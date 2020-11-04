import logging

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import fluid
from paddle.fluid import dygraph
from modules.util import Hourglass, AntiAliasInterpolation2d, kp2gaussian, \
    make_coordinate_grid

# ====================
TEST_MODE = False
if TEST_MODE:
    logging.warning('TEST MODE: dense_motion.py')
# ====================


def _repeat(inTensor, times):
    if not isinstance(inTensor, fluid.core_avx.VarBase):
        raise TypeError('Type of inTensor is :%s' % type(inTensor))
    in_shape = list(inTensor.shape)
    if len(in_shape) == len(times):
        return fluid.layers.expand(inTensor, times)
    else:
        raise TypeError('Repeat in:%s out:%s' % (str(inTensor.shape), str(times)))


class DenseMotionNetwork(paddle.nn.Layer):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """
    
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01, **kwargs):
        super(DenseMotionNetwork, self).__init__(**kwargs)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)
        
        self.mask = dygraph.Conv2D(self.hourglass.out_filters, num_kp + 1, filter_size=(7, 7), padding=(3, 3))
        
        if estimate_occlusion_map:
            self.occlusion = dygraph.Conv2D(self.hourglass.out_filters, 1, filter_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None
        
        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
    
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source
        
        # adding background feature
        zeros = dygraph.to_variable(np.zeros((heatmap.shape[0], 1, spatial_size[0], spatial_size[1]), dtype=np.float32))
        heatmap = fluid.layers.concat([zeros, heatmap], axis=1)
        heatmap = fluid.layers.unsqueeze(heatmap, [2])
        return heatmap
    
    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w))
        identity_grid = identity_grid.reshape((1, 1, h, w, 2))
        _buf = fluid.layers.reshape(kp_driving['value'], (bs, self.num_kp, 1, 1, 2))
        max_shape = np.array([max(i, j) for i, j in zip(identity_grid.shape, _buf.shape)])
        coordinate_grid = fluid.layers.expand(identity_grid, (max_shape / np.array(identity_grid.shape)).astype(np.uint8).tolist()) - fluid.layers.expand(_buf,(max_shape / np.array(_buf.shape)).astype(np.uint8).tolist())
        
        if 'jacobian' in kp_driving:
            jacobian = fluid.layers.matmul(kp_source['jacobian'], paddle.inverse(kp_driving['jacobian']))
            dim_1, dim_2, *else_dim = jacobian.shape
            jacobian = fluid.layers.reshape(jacobian, (-1, *else_dim))
            jacobian = fluid.layers.unsqueeze(fluid.layers.unsqueeze(jacobian, [-3]), [-3])
            jacobian = _repeat(jacobian, (1, h, w, 1, 1))
            
            _, _, *dimm = coordinate_grid.shape
            coordinate_grid = fluid.layers.reshape(coordinate_grid, (-1, *dimm))
            _right = fluid.layers.unsqueeze(coordinate_grid, [-1])
            coordinate_grid = fluid.layers.matmul(jacobian, _right)
            coordinate_grid = fluid.layers.squeeze(coordinate_grid, [-1])
            coordinate_grid = fluid.layers.reshape(coordinate_grid, (dim_1, dim_2, *(coordinate_grid.shape[1:])))
        
        driving_to_source = coordinate_grid + fluid.layers.reshape(kp_source['value'], (bs, self.num_kp, 1, 1, 2))
        
        # adding background feature
        identity_grid = _repeat(identity_grid, (bs, 1, 1, 1, 1))
        sparse_motions = fluid.layers.concat([identity_grid, driving_to_source], axis=1)
        return sparse_motions
    
    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        # import pdb
        # pdb.set_trace()
        bs, _, h, w = source_image.shape
        source_image = fluid.layers.reshape(source_image, (-1, h, w))
        source_repeat = _repeat(fluid.layers.unsqueeze(fluid.layers.unsqueeze(source_image, [1]), [1]),
                                (self.num_kp + 1, 1, 1, 1, 1))
        source_repeat = fluid.layers.reshape(source_repeat, (bs * (self.num_kp + 1), -1, h, w))
        sparse_motions = fluid.layers.reshape(sparse_motions, (bs * (self.num_kp + 1), h, w, -1))
        # TODO: fluid.layers.grid_sampler align_corners==True!!!
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, mode='bilinear', padding_mode='zeros', align_corners=True)
        # if TEST_MODE:
        #     import pdb;pdb.set_trace()
        #     logging.warning('Check')
        sparse_deformed = fluid.layers.reshape(sparse_deformed, (bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed
    
    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
        
        bs, _, h, w = source_image.shape
        
        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source
        
        buf = fluid.layers.concat([heatmap_representation, deformed_source], axis=2)
        buf = fluid.layers.reshape(buf, (bs, -1, h, w))
        
        prediction = self.hourglass(buf)
        
        mask = self.mask(prediction)
        mask = fluid.layers.softmax(mask, axis=1)
        out_dict['mask'] = mask
        mask = fluid.layers.unsqueeze(mask, [2])
        sparse_motion = fluid.layers.transpose(sparse_motion, (0, 1, 4, 2, 3))
        deformation = fluid.layers.reduce_sum(sparse_motion * mask, dim=1)
        deformation = fluid.layers.transpose(deformation, (0, 2, 3, 1))
        
        out_dict['deformation'] = deformation
        
        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = fluid.layers.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map
        return out_dict