import logging
import os
from pathlib import Path
import sys
from argparse import ArgumentParser

import imageio
import numpy as np
import paddle
import yaml
from frames_dataset import FramesDataset, DatasetRepeater, PairedDataset
from modules.discriminator import MultiScaleDiscriminator
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.model import GeneratorFullModel, DiscriminatorFullModel
from paddle import fluid
from paddle.optimizer.lr import MultiStepDecay
from scipy.spatial import ConvexHull
from tqdm import trange

VISUAL = False
TEST_MODE = False
if VISUAL:
    from visualdl import LogWriter
    writer = LogWriter(logdir="./log/fashion/train")
if TEST_MODE:
    logging.warning('TEST MODE: run.py')
    fake_batch_size = 2
    fake_input = np.transpose(np.tile(np.load('/home/aistudio/img.npy')[:1, ...], (fake_batch_size, 1, 1, 1)).astype(np.float32)/255, (0, 3, 1, 2))  #Shape:[fake_batch_size, 3, 256, 256]


def load_ckpt(ckpt_config, generator=None, optimizer_generator=None, kp_detector=None, optimizer_kp_detector=None,
              discriminator=None, optimizer_discriminator=None):
    has_key = lambda key: key in ckpt_config.keys() and ckpt_config[key] is not None
    new_dict = lambda name, valu: dict([(k2, v1) for (k1, v1), (k2, v2) in zip(valu.items(), name.items())])
    if has_key('generator') and generator is not None:
        if ckpt_config['generator'][-3:] == 'npz':
            G_param = np.load(ckpt_config['generator'], allow_pickle=True)['arr_0'].item()
            G_param_clean = dict([(i, G_param[i]) for i in G_param if 'num_batches_tracked' not in i])
            diff_num = np.array([list(i.shape) != list(j.shape) for i, j in
                                 zip(generator.state_dict().values(), G_param_clean.values())]).sum()
            if diff_num == 0:
                generator.set_state_dict(new_dict(generator.state_dict(), G_param_clean))
                logging.info('G is loaded from *.npz')
            else:
                logging.warning('G cannot load from *.npz')
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
            KD_param_clean = dict([(i, KD_param[i]) for i in KD_param if 'num_batches_tracked' not in i])
            diff_num = np.array([list(i.shape) != list(j.shape) for i, j in
                                 zip(kp_detector.state_dict().values(), KD_param_clean.values())]).sum()
            if diff_num == 0:
                kp_detector.set_state_dict(new_dict(kp_detector.state_dict(), KD_param_clean))
                logging.info('KP is loaded from *.npz')
            else:
                logging.warning('KP cannot load from *.npz')
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
                # resort
                ## PP:        [conv.weight,   conv.bias,          weight_u, weight_v]
                ## pytorch:   [conv.bias,     conv.weight_orig,   weight_u, weight_v]
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


def train(config, generator, discriminator, kp_detector, save_dir: Path, dataset):

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
        logging.warning('TEST MODE: Optimer is SGD, lr is 0.001. run.py: L50')
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
    
    # dataset
    if not TEST_MODE:
        dataloader = paddle.io.DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, drop_last=False, num_workers=4, use_buffer_reader=True, use_shared_memory=False)
    
    # load checkpoint
    ckpt_config = config['ckpt_model']
    has_key = lambda key: key in ckpt_config.keys() and ckpt_config[key] is not None
    load_ckpt(ckpt_config, generator, optimizer_generator, kp_detector, optimizer_kp_detector, discriminator, optimizer_discriminator)
    
    # create full model
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    
    # load vgg19
    if has_key('vgg19_model'):
        vggVarList = [i for i in generator_full.vgg.parameters()]
        paramset = np.load(ckpt_config['vgg19_model'], allow_pickle=True)['arr_0']
        for var, v in zip(vggVarList, paramset):
            if list(var.shape) == list(v.shape):
                var.set_value(v)
            else:
                logging.warning('VGG19 cannot be loaded')
        logging.info('Pre-trained VGG19 is loaded from *.npz')

    # train
    generator_full.train()
    discriminator_full.train()
    global_step = 0
    for epoch in trange(start_epoch, train_params['num_epochs']):
        for _step, _x in enumerate(dataloader() if not TEST_MODE else range(100)):
            
            # prepare data
            x = dict()
            if TEST_MODE:
                logging.warning('TEST MODE: Input is Fixed run.py: L207')
                x['driving'] = paddle.to_tensor(fake_input)
                x['source'] = paddle.to_tensor(fake_input)
                x['name'] = ['test1', 'test2']
            else:
                x['driving'], x['source'] = _x
                x['name'] = ['NULL'] * _x[0].shape[0]
            
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
            
            if VISUAL and _step % 3 == 0:
                writer.add_scalar(tag="lr", step=global_step, value=optimizer_generator.get_lr())
                for k, v in losses.items():
                    writer.add_scalar(tag=k, step=global_step, value=v)
            
            # print log
            if _step % 20 == 0:
                logging.info('Epoch:%i\tstep: %i\tLr:%1.7f' % (epoch, _step, optimizer_generator.get_lr()))
                logging.info('\t'.join(['%s:%1.4f' % (k, v) for k, v in losses.items()]))

            global_step += 1
        
        # save
        if epoch % 3 == 0:
            paddle.fluid.save_dygraph(generator.state_dict(), save_dir.joinpath('epoch%i/G' % epoch).__str__())
            paddle.fluid.save_dygraph(discriminator.state_dict(), save_dir.joinpath('epoch%i/D' % epoch).__str__())
            paddle.fluid.save_dygraph(kp_detector.state_dict(), save_dir.joinpath('epoch%i/KP' % epoch).__str__())
            paddle.fluid.save_dygraph(optimizer_generator.state_dict(), save_dir.joinpath('epoch%i/G' % epoch).__str__())
            paddle.fluid.save_dygraph(optimizer_discriminator.state_dict(), save_dir.joinpath('epoch%i/D' % epoch).__str__())
            paddle.fluid.save_dygraph(optimizer_kp_detector.state_dict(), save_dir.joinpath('epoch%i/KP' % epoch).__str__())
            logging.info('Model is saved to:%s' % save_dir.joinpath('epoch%i/' % epoch))
        gen_lr.step()
        dis_lr.step()
        kp_lr.step()

        # eval model
        generator_full.eval()
        discriminator_full.eval()
        kp_detector.eval()
        generator.eval()
        with paddle.no_grad():
            full_video_source = np.stack([imageio.imread(i).astype(np.float32) / 255 for i in sorted(
                list(dataset.dataset.videos[0].iterdir()))])
            full_video = paddle.to_tensor(np.transpose(full_video_source, (3, 0, 1, 2))[np.newaxis, ...])
            predictions = []
            loss_list = []
            kp_source = kp_detector(full_video[:, :, 0])
            for frame_idx in range(full_video.shape[2]):
                source = full_video[:, :, 0]
                driving = full_video[:, :, frame_idx]
                kp_driving = kp_detector(driving)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                del out['sparse_deformed']
        
                out_img = out['prediction'].detach().numpy()
                img = (np.transpose(out_img, [0, 2, 3, 1])[0] * 255).astype(np.uint8)
                predictions.append(img)
                loss_list.append(np.abs(out_img - driving.numpy()).mean())
            orig = (np.concatenate(full_video_source, axis=1) * 255).astype(np.uint8)
            predictions = np.concatenate(predictions, axis=1)
            predictions = np.concatenate([orig, predictions], axis=0)
            imageio.imsave(save_dir.joinpath('epoch%i.png' % epoch), predictions)
            if VISUAL:
                writer.add_image(tag='reconstruction_img', img=predictions, step=global_step)
                writer.add_scalar(tag='reconstruction', step=global_step, value=np.mean(loss_list))
        kp_detector.train()
        generator.train()
        generator_full.train()
        discriminator_full.train()
        


def reconstruction(config, generator, kp_detector, dataset, save_dir=Path('./')):
    png_dir = save_dir.joinpath('reconstruction/png')
    log_dir = save_dir.joinpath('reconstruction')
    ckpt_config = config['ckpt_model']
    load_ckpt(ckpt_config, generator=generator, kp_detector=kp_detector)
    dataloader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, use_buffer_reader=True, use_shared_memory=False)

    if not log_dir.exists(): log_dir.mkdir()
    if not png_dir.exists(): png_dir.mkdir()
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
            imageio.imsave(png_dir.joinpath('%i' % it + '.png'), predictions)
    
            video_cat = np.concatenate([origin_video, visualizations], axis=1)
            image_name = '%i' % it + config['reconstruction_params']['format']
            imageio.mimsave(log_dir.joinpath(image_name), list([i for i in video_cat]))
        bar.update()
    bar.close()
    logging.info("Reconstruction loss: %s" % np.mean(loss_list))


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']
        if use_relative_jacobian:
            jacobian_diff = paddle.matmul(kp_driving['jacobian'], paddle.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = paddle.matmul(jacobian_diff, kp_source['jacobian'])
    return kp_new


def animate(config, generator, kp_detector, dataset, save_dir=Path('./')):
    log_dir = save_dir.joinpath('animation')
    png_dir = save_dir.joinpath('animation/png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, use_buffer_reader=True, use_shared_memory=False)

    ckpt_config = config['ckpt_model']
    load_ckpt(ckpt_config, generator=generator, kp_detector=kp_detector)

    if not log_dir.exists(): log_dir.mkdir()
    if not png_dir.exists(): png_dir.mkdir()

    generator.eval()
    kp_detector.eval()
    bar = trange(animate_params['num_pairs'])
    logging.info('num_pairs: %i' % animate_params['num_pairs'])
    for it, x in enumerate(dataloader):
        bar.update()
        with paddle.no_grad():
            predictions = []
            visualizations = []

            driving_video = x[0]
            source_frame = x[1]

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source
                out['kp_norm'] = kp_norm
                del out['sparse_deformed']
                out_img = out['prediction'].detach().numpy()
                img = (np.transpose(out_img, [0, 2, 3, 1])[0] * 255).astype(np.uint8)
                visualizations.append(img)
                predictions.append(img)

            origin_video = np.transpose(x[0][0].numpy() * 255, (1, 2, 3, 0)).astype(np.uint8)
            visualizations = np.stack(visualizations)
            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(png_dir.joinpath('%i' % it + '.png'), predictions)

            video_cat = np.concatenate([origin_video, visualizations], axis=1)
            image_name = '%i' % it + animate_params['format']
            imageio.mimsave(log_dir.joinpath(image_name), list([i for i in video_cat]))


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
    if not TEST_MODE:
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

    save_dir = Path(opt.save_dir)
    if opt.mode == 'train':
        logging.info("Start training...")
        if TEST_MODE:
            dataset = None
        else:
            dataset = DatasetRepeater(dataset, config['train_params']['num_repeats'])
        train(config, generator, discriminator, kp_detector, save_dir, dataset)
    elif opt.mode == 'reconstruction':
        logging.info("Reconstruction...")
        reconstruction(config, generator, kp_detector, dataset, save_dir=save_dir)
    elif opt.mode == 'animate':
        logging.info("Animate...")
        animate(config, generator, kp_detector, dataset, save_dir=save_dir)