from tqdm import trange
import numpy as np
import os
import paddle
import random
from paddle import fluid
from paddle.fluid import dygraph
if paddle.version.full_version == '1.8.4':
    from paddle.fluid.dygraph.learning_rate_scheduler import MultiStepDecay
elif paddle.version.full_version == '0.0.0':
    from paddle.fluid.dygraph import MultiStepDecay
elif paddle.version.full_version in ['1.8.2', '1.8.0']:
    from modules.MultiStepLR import MultiStepDecay
from modules.model import GeneratorFullModel, DiscriminatorFullModel

def train(config, generator, discriminator, kp_detector, save_dir, dataset):
    train_params = config['train_params']
    # learning_rate_scheduler
    if paddle.version.full_version in ['1.8.4', '0.0.0', '1.8.2', '1.8.0']:
        gen_lr = MultiStepDecay(learning_rate=train_params['lr_generator'], milestones=train_params['epoch_milestones'], decay_rate=0.1)
        dis_lr = MultiStepDecay(learning_rate=train_params['lr_discriminator'], milestones=train_params['epoch_milestones'], decay_rate=0.1)
        kp_lr = MultiStepDecay(learning_rate=train_params['lr_kp_detector'], milestones=train_params['epoch_milestones'], decay_rate=0.1)
    else:
        gen_lr = train_params['lr_generator']
        dis_lr = train_params['lr_discriminator']
        kp_lr = train_params['lr_kp_detector']
    # optimer
    optimizer_generator = fluid.optimizer.AdamOptimizer(
        parameter_list=generator.parameters(),
        learning_rate=gen_lr
    )
    optimizer_discriminator = fluid.optimizer.AdamOptimizer(
        parameter_list=discriminator.parameters(),
        learning_rate=dis_lr
    )
    optimizer_kp_detector = fluid.optimizer.AdamOptimizer(
        parameter_list=kp_detector.parameters(),
        learning_rate=kp_lr
    )
    # load start_epoch
    if isinstance(config['ckpt_model']['start_epoch'], int):
        start_epoch = config['ckpt_model']['start_epoch']
    else:
        start_epoch = 0
    print('start_epoch: %i'%start_epoch)

    # dataset pipeline
    def indexGenertaor():
        """随机生成索引序列
        """
        order = list(range(len(dataset)))
        order = order * train_params['num_repeats']
        random.shuffle(order)
        for i in order:
            yield i
    _dataset = fluid.io.xmap_readers(dataset.getSample, indexGenertaor, process_num=4, buffer_size=128, order=False)
    _dataset = fluid.io.batch(_dataset, batch_size=train_params['batch_size'], drop_last=True)
    dataloader = fluid.io.buffered(_dataset, 1)

    # model
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    ###### Restore Part ######
    ckpt_config = config['ckpt_model']
    has_key = lambda key: key in ckpt_config.keys() and ckpt_config[key] is not None
    if has_key('vgg19_model'):
        vggVarList = [i for i in generator_full.vgg.parameters()][2:]
        paramset = np.load(ckpt_config['vgg19_model'], allow_pickle=True)['arr_0']
        for var, v in zip(vggVarList, paramset):
            if list(var.shape) == list(v.shape):
                var.set_value(v)
            else:
                print('Restore Error')
        print('Restore Pre-trained VGG19 from npz')
    if has_key('generator'):
        if ckpt_config['generator'][-3:] == 'npz':
            G_param = np.load(ckpt_config['generator'], allow_pickle=True)['arr_0'].item()
            G_param_clean = [(i, G_param[i]) for i in G_param if 'num_batches_tracked' not in i]
            parameter_clean = generator.parameters()
            del(parameter_clean[65])  # The parameters in AntiAliasInterpolation2d is not in dict_set and should be ignore.
            for v, b in zip(parameter_clean, G_param_clean):
                v.set_value(b[1])
            print('Restore Generator from NPZ')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['generator'])
            generator.set_dict(param)
            if optim is not None:
                optimizer_generator.set_dict(optim)
            else:
                print('optimizer of G is not loaded')
            print('Restore Generator from Pdparams')
    if has_key('kp'):
        if ckpt_config['kp'][-3:] == 'npz':
            KD_param = np.load(ckpt_config['kp'], allow_pickle=True)['arr_0'].item()
            KD_param_clean = [(i, KD_param[i]) for i in KD_param if 'num_batches_tracked' not in i]
            parameter_cleans = kp_detector.parameters()
            for v, b in zip(parameter_cleans, KD_param_clean):
                v.set_value(b[1])
            print('Restore KP from NPZ')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['kp'])
            kp_detector.set_dict(param)
            if optim is not None:
                optimizer_kp_detector.set_dict(optim)
            else:
                print('optimizer of KP is not loaded')
            print('Restore KP from Pdparams')
    if has_key('discriminator'):
        if ckpt_config['discriminator'][-3:] == 'npz':
            D_param = np.load(ckpt_config['discriminator'], allow_pickle=True)['arr_0'].item()
            if 'NULL Place' in ckpt_config['discriminator']:
                # 针对未开启spectral_norm的Fashion数据集模型
                ## fashion数据集的默认设置中未启用spectral_norm，但其官方ckpt文件中存在spectral_norm特有的参数 需要重排顺序
                ## 已提相关issue，作者回应加了sn也没什么影响 https://github.com/AliaksandrSiarohin/first-order-model/issues/264
                ## 若在配置文件中开启sn则可通过else语句中的常规方法读取，故现已在配置中开启sn。
                D_param_clean = [(i, D_param[i]) for i in D_param if 'num_batches_tracked' not in i and 'weight_v' not in i and 'weight_u' not in i]
                for idx in range(len(D_param_clean)//2):
                    if 'conv.bias' in D_param_clean[idx*2][0]:
                        D_param_clean[idx*2], D_param_clean[idx*2+1] = D_param_clean[idx*2+1], D_param_clean[idx*2]
                parameter_clean = discriminator.parameters()
                for v, b in zip(parameter_clean, D_param_clean):
                    v.set_value(b[1])
            else:
                D_param_clean = list(D_param.items())
                parameter_clean = discriminator.parameters()
                assert len(D_param_clean) == len(parameter_clean)
                # 调换顺序
                ## PP中:        [conv.weight,   conv.bias,          weight_u, weight_v]
                ## pytorch中:   [conv.bias,     conv.weight_orig,   weight_u, weight_v]
                for idx in range(len(parameter_clean)):
                    if list(parameter_clean[idx].shape) == list(D_param_clean[idx][1].shape):
                        parameter_clean[idx].set_value(D_param_clean[idx][1])
                    elif parameter_clean[idx].name.split('.')[-1] == 'w_0' and D_param_clean[idx+1][0].split('.')[-1] == 'weight_orig':
                        parameter_clean[idx].set_value(D_param_clean[idx+1][1])
                    elif parameter_clean[idx].name.split('.')[-1] == 'b_0' and D_param_clean[idx-1][0].split('.')[-1] == 'bias':
                        parameter_clean[idx].set_value(D_param_clean[idx-1][1])
                    else:
                        print('Error', idx)
            print('Restore Discriminator from NPZ')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['discriminator'])
            discriminator.set_dict(param)
            if optim is not None:
                optimizer_discriminator.set_dict(optim)
            else:
                print('optimizer of Discriminator is not loaded')
            print('Restore Discriminator from Pdparams')
    ###### Restore Part END ######

    generator_full.train()
    discriminator_full.train()
    for epoch in trange(start_epoch, train_params['num_epochs']):
        for _step, _x in enumerate(dataloader()):
            x = dict()
            for _key in _x[0].keys():
                if str(_key) != 'name':
                    x[_key] = dygraph.to_variable(np.stack([_v[_key] for _v in _x], axis=0).astype(np.float32))
                else:
                    x[_key] = np.stack([_v[_key] for _v in _x], axis=0)
            # train generator
            losses_generator, generated = generator_full(x.copy())
            loss_values = [fluid.layers.reduce_sum(val) for val in losses_generator.values()]
            loss = fluid.layers.sum(loss_values)
            loss.backward()
            optimizer_generator.minimize(loss)
            optimizer_generator.clear_gradients()
            optimizer_kp_detector.minimize(loss)
            optimizer_kp_detector.clear_gradients()

            # train discriminator
            if train_params['loss_weights']['generator_gan'] != 0:
                optimizer_discriminator.clear_gradients()
                losses_discriminator = discriminator_full(x.copy(), generated)
                loss_values = [fluid.layers.reduce_mean(val) for val in losses_discriminator.values()]
                loss = fluid.layers.sum(loss_values)
                loss.backward()
                optimizer_discriminator.minimize(loss)
                optimizer_discriminator.clear_gradients()
            else:
                losses_discriminator = {}

            losses_generator.update(losses_discriminator)
            losses = {key: fluid.layers.reduce_mean(value).detach().numpy() for key, value in losses_generator.items()}

            # print log
            if _step % 20 == 0:
                print('Epoch:%i\tstep: %i\tLr:%1.7f'%(epoch, _step, optimizer_generator.current_step_lr()))
                print('\t'.join(['%s:%1.4f'%(k,v) for k,v in losses.items()]))

        # save
        if epoch%3 == 0:
            paddle.fluid.save_dygraph(generator.state_dict(), os.path.join(save_dir, 'epoch%i/G'%epoch))
            paddle.fluid.save_dygraph(discriminator.state_dict(), os.path.join(save_dir, 'epoch%i/D'%epoch))
            paddle.fluid.save_dygraph(kp_detector.state_dict(), os.path.join(save_dir, 'epoch%i/KP'%epoch))
            paddle.fluid.save_dygraph(optimizer_generator.state_dict(), os.path.join(save_dir, 'epoch%i/G'%epoch))
            paddle.fluid.save_dygraph(optimizer_discriminator.state_dict(), os.path.join(save_dir, 'epoch%i/D'%epoch))
            paddle.fluid.save_dygraph(optimizer_kp_detector.state_dict(), os.path.join(save_dir, 'epoch%i/KP'%epoch))
        if paddle.version.full_version in ['1.8.4', '0.0.0', '1.8.2', '1.8.0']:
            gen_lr.epoch()
            dis_lr.epoch()
            kp_lr.epoch()
