import imageio
import logging
import numpy as np
import os
import paddle
import sys
import yaml
from argparse import ArgumentParser
from frames_dataset import FramesDataset, DatasetRepeater
from modules.discriminator import MultiScaleDiscriminator
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.model import GeneratorFullModel, DiscriminatorFullModel
from paddle import fluid
from paddle.optimizer.lr import MultiStepDecay
from tqdm import trange

# from reconstruction import reconstruction
# from animate import animate

TEST_MODE = False
if TEST_MODE:
    logging.warning('TEST MODE: train.py')
    # fake_input可随意指定,此处的batchsize=2
    fake_input = np.transpose(np.tile(np.load('/home/aistudio/img.npy')[:1, ...], (2, 1, 1, 1)).astype(np.float32)/255, (0, 3, 1, 2))  #Shape:[2, 3, 256, 256]


def load_ckpt(ckpt_config, generator=None, optimizer_generator=None, kp_detector=None, optimizer_kp_detector=None,
              discriminator=None, optimizer_discriminator=None):
    has_key = lambda key: key in ckpt_config.keys() and ckpt_config[key] is not None
    if has_key('generator') and generator is not None:
        if ckpt_config['generator'][-3:] == 'npz':
            G_param = np.load(ckpt_config['generator'], allow_pickle=True)['arr_0'].item()
            G_param_clean = dict([(i, G_param[i]) for i in G_param if 'num_batches_tracked' not in i])
            diff_num = np.array([list(i.shape) != list(j.shape) for i, j in
                                 zip(generator.state_dict().values(), G_param_clean.values())]).sum()
            if diff_num == 0:
                # rename key
                assign_dict = dict(
                    [(i[0], j[1]) for i, j in zip(generator.state_dict().items(), G_param_clean.items())])
                # TODO: try generator.set_state_dict(G_param_clean, use_structured_name=False)
                generator.set_state_dict(assign_dict, use_structured_name=False)
                logging.info('Generator is loaded from *.npz')
            else:
                logging.warning('Generator cannot load from *.npz')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['generator'])
            generator.set_dict(param)
            if optim is not None and optimizer_generator is not None:
                optimizer_generator.set_state_dict(optim)
            else:
                logging.info('Optimizer of G is not loaded')
            logging.info('Generator is loaded from *.pdparams')
    if has_key('kp') and kp_detector is not None:
        if ckpt_config['kp'][-3:] == 'npz':
            KD_param = np.load(ckpt_config['kp'], allow_pickle=True)['arr_0'].item()
            KD_param_clean = [(i, KD_param[i]) for i in KD_param if 'num_batches_tracked' not in i]
            parameter_cleans = kp_detector.parameters()
            # TODO: try kp_detector.set_state_dict(KD_param_clean, use_structured_name=False)
            for v, b in zip(parameter_cleans, KD_param_clean):
                v.set_value(b[1])
            logging.info('KP is loaded from *.npz')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['kp'])
            kp_detector.set_dict(param)
            if optim is not None and optimizer_kp_detector is not None:
                optimizer_kp_detector.set_state_dict(optim)
            else:
                logging.info('Optimizer of KP is not loaded')
            logging.info('KP is loaded from *.pdparams')
    if has_key('discriminator') and discriminator is not None:
        if ckpt_config['discriminator'][-3:] == 'npz':
            D_param = np.load(ckpt_config['discriminator'], allow_pickle=True)['arr_0'].item()
            if 'NULL Place' in ckpt_config['discriminator']:
                # 针对未开启spectral_norm的Fashion数据集模型
                ## fashion数据集的默认设置中未启用spectral_norm，但其官方ckpt文件中存在spectral_norm特有的参数 需要重排顺序
                ## 已提相关issue，作者回应加了sn也没什么影响 https://github.com/AliaksandrSiarohin/first-order-model/issues/264
                ## 若在配置文件中开启sn则可通过else语句中的常规方法读取，故现已在配置中开启sn。
                D_param_clean = [(i, D_param[i]) for i in D_param if
                                 'num_batches_tracked' not in i and 'weight_v' not in i and 'weight_u' not in i]
                for idx in range(len(D_param_clean) // 2):
                    if 'conv.bias' in D_param_clean[idx * 2][0]:
                        D_param_clean[idx * 2], D_param_clean[idx * 2 + 1] = D_param_clean[idx * 2 + 1], \
                                                                             D_param_clean[
                                                                                 idx * 2]
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
                    elif parameter_clean[idx].name.split('.')[-1] == 'w_0' and D_param_clean[idx + 1][0].split('.')[
                        -1] == 'weight_orig':
                        parameter_clean[idx].set_value(D_param_clean[idx + 1][1])
                    elif parameter_clean[idx].name.split('.')[-1] == 'b_0' and D_param_clean[idx - 1][0].split('.')[
                        -1] == 'bias':
                        parameter_clean[idx].set_value(D_param_clean[idx - 1][1])
                    else:
                        logging.error('Error', idx)
            logging.info('Discriminator is loaded from *.npz')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['discriminator'])
            discriminator.set_dict(param)
            if optim is not None and optimizer_discriminator is not None:
                optimizer_discriminator.set_state_dict(optim)
            else:
                logging.info('Optimizer of Discriminator is not loaded')
            logging.info('Discriminator is loaded from *.pdparams')


def train(config, generator, discriminator, kp_detector, save_dir, dataset):
    train_params = config['train_params']
    
    # learning_rate_scheduler
    gen_lr = MultiStepDecay(learning_rate=train_params['lr_generator'], milestones=train_params['epoch_milestones'],
                            gamma=0.1)
    dis_lr = MultiStepDecay(learning_rate=train_params['lr_discriminator'],
                            milestones=train_params['epoch_milestones'], gamma=0.1)
    kp_lr = MultiStepDecay(learning_rate=train_params['lr_kp_detector'],
                           milestones=train_params['epoch_milestones'], gamma=0.1)
    # optimer
    if TEST_MODE:
        logging.warning('TEST MODE: Optimer is SGD, lr is 0.001. train.py: L50')
        optimizer_generator = paddle.optimizer.SGD(
            parameters=generator.parameters(),
            learning_rate=0.001
        )
        optimizer_discriminator = paddle.optimizer.SGD(
            parameters=discriminator.parameters(),
            learning_rate=0.001
        )
        optimizer_kp_detector = paddle.optimizer.SGD(
            parameters=kp_detector.parameters(),
            learning_rate=0.001
        )
    else:
        optimizer_generator = paddle.optimizer.Adam(
            parameters=generator.parameters(),
            learning_rate=gen_lr
        )
        optimizer_discriminator = paddle.optimizer.Adam(
            parameters=discriminator.parameters(),
            learning_rate=dis_lr
        )
        optimizer_kp_detector = paddle.optimizer.Adam(
            parameters=kp_detector.parameters(),
            learning_rate=kp_lr
        )
    
    # load start_epoch
    if isinstance(config['ckpt_model']['start_epoch'], int):
        start_epoch = config['ckpt_model']['start_epoch']
    else:
        start_epoch = 0
    logging.info('Start Epoch is :%i' % start_epoch)
    
    # dataset pipeline
    dataloader = paddle.io.DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, drop_last=False, num_workers=4, use_buffer_reader=True, use_shared_memory=False)

    ###### Restore Part ######
    ckpt_config = config['ckpt_model']
    has_key = lambda key: key in ckpt_config.keys() and ckpt_config[key] is not None
    load_ckpt(ckpt_config, generator, optimizer_generator, kp_detector, optimizer_kp_detector, discriminator, optimizer_discriminator)
    
    # create model
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    if has_key('vgg19_model'):
        vggVarList = [i for i in generator_full.vgg.parameters()]
        paramset = np.load(ckpt_config['vgg19_model'], allow_pickle=True)['arr_0']
        for var, v in zip(vggVarList, paramset):
            if list(var.shape) == list(v.shape):
                var.set_value(v)
            else:
                logging.warning('VGG19 cannot be loaded')
        logging.info('Pre-trained VGG19 is loaded from *.npz')
    generator_full.train()
    discriminator_full.train()
    for epoch in trange(start_epoch, train_params['num_epochs']):
        for _step, _x in enumerate(dataloader()):
            # prepare data
            x = dict()
            x['driving'], x['source'] = _x
            x['name'] = ['NULL'] * _x[0].shape[0]
            if TEST_MODE:
                logging.warning('TEST MODE: Input is Fixed train.py: L207')
                x['driving'] = paddle.to_tensor(fake_input)
                x['source'] = paddle.to_tensor(fake_input)
                x['name'] = ['test1', 'test2']
            # train generator
            losses_generator, generated = generator_full(x.copy())
            loss_values = [val.sum() for val in losses_generator.values()]
            loss = paddle.add_n(loss_values)
            if TEST_MODE:
                print('Check Generator Loss')
                print('\n'.join(['%s:%1.5f'%(k,v.numpy()) for k,v in zip(losses_generator.keys(), loss_values)]))
                import pdb;pdb.set_trace();
            loss.backward()
            optimizer_generator.step()
            optimizer_generator.clear_grad()
            optimizer_kp_detector.step()
            optimizer_kp_detector.clear_grad()
            
            # train discriminator
            if train_params['loss_weights']['generator_gan'] != 0:
                optimizer_discriminator.clear_gradients()
                losses_discriminator = discriminator_full(x.copy(), generated)
                loss_values = [val.mean() for val in losses_discriminator.values()]
                loss = paddle.add_n(loss_values)
                if TEST_MODE:
                    print('Check Discriminator Loss')
                    print('\n'.join(['%s:%1.5f'%(k,v.numpy()) for k,v in zip(losses_discriminator.keys(), loss_values)]))
                    import pdb;pdb.set_trace();
                loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.clear_grad()
            else:
                losses_discriminator = {}
            
            losses_generator.update(losses_discriminator)
            losses = {key: value.mean().detach().numpy() for key, value in losses_generator.items()}
            
            # print log
            if _step % 20 == 0:
                logging.info('Epoch:%i\tstep: %i\tLr:%1.7f' % (epoch, _step, optimizer_generator.get_lr()))
                logging.info('\t'.join(['%s:%1.4f' % (k, v) for k, v in losses.items()]))
        
        # save
        if epoch % 3 == 0:
            paddle.fluid.save_dygraph(generator.state_dict(), os.path.join(save_dir, 'epoch%i/G' % epoch))
            paddle.fluid.save_dygraph(discriminator.state_dict(), os.path.join(save_dir, 'epoch%i/D' % epoch))
            paddle.fluid.save_dygraph(kp_detector.state_dict(), os.path.join(save_dir, 'epoch%i/KP' % epoch))
            paddle.fluid.save_dygraph(optimizer_generator.state_dict(), os.path.join(save_dir, 'epoch%i/G' % epoch))
            paddle.fluid.save_dygraph(optimizer_discriminator.state_dict(), os.path.join(save_dir, 'epoch%i/D' % epoch))
            paddle.fluid.save_dygraph(optimizer_kp_detector.state_dict(), os.path.join(save_dir, 'epoch%i/KP' % epoch))
            logging.info('Model is saved to:%s' % os.path.join(save_dir, 'epoch%i/' % epoch))
        gen_lr.step()
        dis_lr.step()
        kp_lr.step()


def reconstruction(config, generator, kp_detector, dataset, save_dir='./'):
    png_dir = os.path.join(save_dir, 'reconstruction/png')
    log_dir = os.path.join(save_dir, 'reconstruction')
    ckpt_config = config['ckpt_model']
    load_ckpt(ckpt_config, generator=generator, kp_detector=kp_detector)
    dataloader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, use_buffer_reader=True, use_shared_memory=False)

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(png_dir): os.makedirs(png_dir)
    loss_list = []
    generator.eval()
    kp_detector.eval()

    bar = trange(config['reconstruction_params']['num_videos'])
    logging.info('num_videos: %i' % config['reconstruction_params']['num_videos'])
    for it, x in enumerate(dataloader()):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with paddle.no_grad():
            predictions = []
            visualizations = []
            kp_source = kp_detector(x[0][:, :, 0])
            for frame_idx in range(x[0].shape[2]):
                source = x[0][:, :, 0]
                driving = x[0][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                del out['sparse_deformed']
        
                out_img = out['prediction'].detach().numpy()
                img = (np.transpose(out_img, [0, 2, 3, 1])[0] * 255).astype(np.uint8)
                visualizations.append(img)
                predictions.append(img)
                loss_list.append(np.abs(out_img - driving.numpy()).mean())
            origin_video = np.transpose(x[0][0].numpy() * 255, (1, 2, 3, 0)).astype(np.uint8)
            visualizations = np.stack(visualizations)
            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, '%i' % it + '.png'), predictions)
    
            video_cat = np.concatenate([origin_video, visualizations], axis=1)
            image_name = '%i' % it + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), list([i for i in video_cat]))
        bar.update()
    bar.close()
    logging.info("Reconstruction loss: %s" % np.mean(loss_list))


if __name__ == "__main__":
    paddle.set_device("gpu")
    logging.getLogger().setLevel(logging.INFO)
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--save_dir", default='/home/aistudio/train_ckpt', help="path to save in")
    parser.add_argument("--preload", action='store_true', help="preload dataset to RAM")
    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'], **config['model_params']['common_params'])
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'], **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'], **config['model_params']['common_params'])

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    if opt.preload:
        logging.info('PreLoad Dataset: Start')
        pre_list = list(range(len(dataset)))
        import multiprocessing.pool as pool
        with pool.Pool(4) as pl:
            buf = pl.map(dataset.preload, pre_list)
        for idx, (i,v) in enumerate(zip(pre_list, buf)):
            dataset.buffed[i] = v.copy()
            buf[idx] = None
        logging.info('PreLoad Dataset: End')

    if opt.mode == 'train':
        save_dir = opt.save_dir
        logging.info("Start training...")
        dataset = DatasetRepeater(dataset, config['train_params']['num_repeats'])
        train(config, generator, discriminator, kp_detector, save_dir, dataset)
    elif opt.mode == 'reconstruction':
        logging.info("Reconstruction...")
        reconstruction(config, generator, kp_detector, dataset)
    # elif opt.mode == 'animate':
    #     print("Animate...")
    #     animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)