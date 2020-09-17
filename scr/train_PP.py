# import pdb
import matplotlib
from paddle import fluid
import paddle.fluid.dygraph as dygraph
# matplotlib.use('Agg')
import os, sys
import yaml
import numpy as np
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
plac = fluid.CUDAPlace(0)
from train import train
# from reconstruction import reconstruction
# from animate import animate

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    # parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--save_dir", default='/home/aistudio/train_ckpt', help="path to save in")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--preload", action='store_true', help="preload dataset to RAM")
    parser.set_defaults(verbose=False)
    # opt = parser.parse_args(args=['--config', './config/mgif-256.yaml'])
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)
    save_dir = opt.save_dir
    with dygraph.guard(plac):
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])

    if opt.verbose:
        print(generator)
    with dygraph.guard(plac):
        discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                **config['model_params']['common_params'])

    if opt.verbose:
        print(discriminator)
    with dygraph.guard(plac):
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])

    if opt.verbose:
        print(kp_detector)
    if 'hdf5' in config['dataset_params']['root_dir'].lower():
        print('HDF5 Dataset')
        from h5dat import FramesDataset as FramesDataset
    else:
        from frames_dataset import FramesDataset as FramesDataset
    
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    # pre_list = np.random.choice(len(dataset), replace=False, size=len(dataset)//4)
    if opt.preload:
        print('PreLoad Start')
        pre_list = list(range(len(dataset)))
        # HDF5 不可多进程读取
        if 'hdf5' in config['dataset_params']['root_dir'].lower():
            for i in pre_list:
                dataset.buffed[i] = dataset.preload(i)
                if i%100==0:
                    print('Loaded:%i'%i)
        else:
            import multiprocessing.pool as pool
            with pool.Pool(4) as pl:
                buf = pl.map(dataset.preload, pre_list)
            for idx, (i,v) in enumerate(zip(pre_list, buf)):
                dataset.buffed[i] = v.copy()
                buf[idx] = None
                if idx%100==0:
                    print('Loaded:%i'%idx)
        print('PreLoad End')
    else:
        print('Not PreLoad')

    if opt.mode == 'train':
        print("Training...")
        with dygraph.guard(plac):
            train(config, generator, discriminator, kp_detector, save_dir, dataset)
    # elif opt.mode == 'reconstruction':
    #     print("Reconstruction...")
    #     reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    # elif opt.mode == 'animate':
    #     print("Animate...")
    #     animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)