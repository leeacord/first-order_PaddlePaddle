import logging

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid

TEST_MODE = False
if TEST_MODE:
    logging.warning('TEST MODE: model.py')


class conv_block(paddle.nn.Layer):
    def __init__(self, input_channels, num_filter, groups, name=None, use_bias=False):
        super(conv_block, self).__init__()
        self._layers = []
        i = 0
        # 'act' is not a parameter of nn.Conv2D
        self.conv_in = paddle.fluid.dygraph.Conv2D(
            input_channels,
            num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            param_attr=paddle.ParamAttr(name=name + str(i + 1) + "_weights"),
            bias_attr=False if not use_bias else paddle.ParamAttr(name=name + str(i + 1) + "_bias")
        )
        if groups == 1:
            return
        for i in range(1, groups):
            _a = paddle.fluid.dygraph.Conv2D(
                num_filter,
                num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=paddle.ParamAttr(name=name + str(i + 1) + "_weights"),
                bias_attr=False if not use_bias else paddle.ParamAttr(name=name + str(i + 1) + "_bias")
            )
            self._layers.append(_a)
        self.conv = nn.Sequential(*self._layers)
    
    def forward(self, x):
        feat = self.conv_in(x)
        out = F.max_pool2d(self.conv(feat), kernel_size=2, stride=2)
        return out, feat


class Vgg19(paddle.nn.Layer):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    
    def __init__(self, layers=19, class_dim=1000, torch_version=True, requires_grad=False):
        super(Vgg19, self).__init__()
        self.layers = layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)
        
        nums = vgg_spec[layers]
        self.conv1 = conv_block(3, 64, nums[0], name="conv1_", use_bias=True if torch_version else False)
        self.conv2 = conv_block(64, 128, nums[1], name="conv2_", use_bias=True if torch_version else False)
        self.conv3 = conv_block(128, 256, nums[2], name="conv3_", use_bias=True if torch_version else False)
        self.conv4 = conv_block(256, 512, nums[3], name="conv4_", use_bias=True if torch_version else False)
        self.conv5 = conv_block(512, 512, nums[4], name="conv5_", use_bias=True if torch_version else False)
        self.mean = paddle.to_tensor(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32))
        self.std = paddle.to_tensor(np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32))
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        feat, feat_1 = self.conv1(x)
        feat, feat_2 = self.conv2(feat)
        feat, feat_3 = self.conv3(feat)
        feat, feat_4 = self.conv4(feat)
        _, feat_5 = self.conv5(feat)
        return [feat_1, feat_2, feat_3, feat_4, feat_5]


class ImagePyramide(paddle.nn.Layer):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        self.downs = paddle.nn.LayerList()
        self.name_list = []
        for scale in scales:
            self.downs.add_sublayer(str(scale).replace('.', '-'), AntiAliasInterpolation2d(num_channels, scale))
            self.name_list.append(str(scale).replace('.', '-'))
    
    def forward(self, x):
        out_dict = {}
        for scale, down_module in zip(self.name_list, self.downs):
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    
    def __init__(self, bs, **kwargs):
        # noise = np.random.normal(loc=0, scale=kwargs['sigma_affine'], size=(bs, 2, 3))
        noise = paddle.distribution.Normal(loc=[0], scale=[kwargs['sigma_affine']]).sample([bs, 2, 3])
        noise = noise.reshape((bs, 2, 3))
        if TEST_MODE:
            noise = paddle.to_tensor(np.ones((bs, 2, 3)).astype(np.float32))
        
        self.theta = noise + paddle.tensor.eye(2, 3, dtype='float32').reshape((1, 2, 3))
        self.bs = bs
        
        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps'])).unsqueeze(0)
            if TEST_MODE:
                self.control_params = paddle.to_tensor(np.ones((bs, 1, kwargs['points_tps'] ** 2)).astype(np.float32))
            else:
                buf = paddle.distribution.Normal(loc=[0], scale=[kwargs['sigma_tps']]).sample(
                    [bs, 1, kwargs['points_tps'] ** 2])
                self.control_params = buf.reshape((bs, 1, kwargs['points_tps'] ** 2))
        else:
            self.tps = False
    
    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], 'float32').unsqueeze(0)
        grid = grid.reshape((1, frame.shape[2] * frame.shape[3], 2))
        grid = self.warp_coordinates(grid).reshape((self.bs, frame.shape[2], frame.shape[3], 2))
        return F.grid_sample(frame, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    
    def warp_coordinates(self, coordinates):
        theta = self.theta.astype('float32')
        theta = theta.unsqueeze(1)
        coordinates = coordinates.unsqueeze(-1)
        # If x1:(1, 5, 2, 2), x2:(10, 100, 2, 1)
        # torch.matmul can broadcast x1, x2 to (10, 100, ...)
        # In PDPD, it should be done manually
        theta_part_a = theta[:, :, :, :2]
        theta_part_b = theta[:, :, :, 2:]
        # TODO: paddle.matmul have no double_grad_op, use 'paddle.fluid.layers.matmul'
        transformed = paddle.fluid.layers.matmul(*broadcast_v1(theta_part_a, coordinates)) + theta_part_b
        transformed = transformed.squeeze(-1)
        if self.tps:
            control_points = self.control_points.astype('float32')
            control_params = self.control_params.astype('float32')
            distances = coordinates.reshape((coordinates.shape[0], -1, 1, 2)) - control_points.reshape((1, 1, -1, 2))
            distances = distances.abs().sum(-1)
    
            result = distances * distances
            result = result * paddle.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(2).reshape((self.bs, coordinates.shape[1], 1))
            transformed = transformed + result
        return transformed
    
    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)  # When batch_size is 5, the shape of coordinates and new_coordinates is (5, 10, 2)
        # PDPD cannot use new_coordinates[..., 0]
        assert len(new_coordinates.shape) == 3
        grad_x = paddle.grad(new_coordinates[:, :, 0].sum(), coordinates, create_graph=True)
        grad_y = paddle.grad(new_coordinates[:, :, 1].sum(), coordinates, create_graph=True)
        jacobian = paddle.concat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], axis=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(paddle.nn.Layer):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """
    
    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.loss_weights = train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
    
    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        
        loss_values = {}
        
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        
        # VGG19 perceptual Loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = paddle.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total
        
        # Generator Loss
        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total
            
            # Feature matching Loss
            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = paddle.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                loss_values['feature_matching'] = value_total
        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)
            
            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
            
            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = paddle.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = paddle.matmul(
                    *broadcast_v1(transform.jacobian(transformed_kp['value']), transformed_kp['jacobian']))
                normed_driving = paddle.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = paddle.matmul(*broadcast_v1(normed_driving, normed_transformed))
                eye = paddle.tensor.eye(2, dtype='float32').reshape((1, 1, 2, 2))
                
                value = paddle.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
        return loss_values, generated


class DiscriminatorFullModel(paddle.nn.Layer):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """
    
    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.loss_weights = train_params['loss_weights']
    
    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())
        
        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
        
        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total
        
        return loss_values


def broadcast_v1(x, y):
    """
    Broadcast before matmul
    """
    if len(x.shape) != len(y.shape):
        print(x.shape, '!=', y.shape)
        raise ValueError()
    *dim_x, _, _ = x.shape
    *dim_y, _, _ = y.shape
    max_shape = np.max(np.stack([dim_x, dim_y], axis=0), axis=0)
    x_bc = paddle.broadcast_to(x, (*max_shape, x.shape[-2], x.shape[-1]))
    y_bc = paddle.broadcast_to(y, (*max_shape, y.shape[-2], y.shape[-1]))
    return x_bc, y_bc
