import glob
import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
from imageio import imread, mimread, imwrite
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from paddle.io import Dataset
from skimage import io
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F
from paddle.vision.transforms.transforms import _check_input
import random


def read_video(name: pathlib.Path, frame_shape, saveto='folder'):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """
    if name.is_dir():
        frames = sorted(name.iterdir(), key=lambda x: int(x.with_suffix('').name))
        video_array = np.asarray([imread(path)[:, :, :3] for path in frames])
    elif name.suffix.lower() in ['.gif', '.mp4', '.mov']:
        try:
            video = mimread(name)
        except Exception as err:
            logging.error('DataLoading File:%s Msg:%s' % (str(name), str(err)))
            return None

        # convert to 3-channel image
        if video[0].shape[-1] == 4:
            video = [i[..., :3] for i in video]
        elif video[0].shape[-1] == 1:
            video = [np.tile(i, (1, 1, 3)) for i in video]
        elif len(video[0].shape) == 2:
            video = [np.tile(i[..., np.newaxis], (1, 1, 3)) for i in video]
        video_array = np.asarray(video)
        if saveto == 'folder':
            sub_dir = name.with_suffix('')
            try:
                sub_dir.mkdir()
            except FileExistsError:
                pass
            for idx, img in enumerate(video_array):
                imwrite(sub_dir.joinpath('%i.png' % idx), img)
            name.unlink()
    else:
        raise Exception("Unknown dataset file extensions  %s" % name)
    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    FramesDataset[i]: obtain sample from i-th video in self.videos
    """
    
    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, create_frames_folder=True):
        self.root_dir = pathlib.Path(root_dir)
        self.videos = None
        self.frame_shape = frame_shape
        self.id_sampling = id_sampling
        self.is_train = is_train
        self.pairs_list = pairs_list
        self.create_frames_folder = create_frames_folder
        self.augmentation_params = augmentation_params
        self.time_flip = augmentation_params['flip_param']['time_flip']
        self.transform = None
        if self.root_dir.joinpath('train').exists():
            assert self.root_dir.joinpath('test').exists()
            logging.info("Use predefined train-test split.")
            if self.id_sampling:
                train_videos = {video.name.split('#')[0] for video in
                                self.root_dir.joinpath('train').iterdir()}
                train_videos = list(train_videos)
            else:
                train_videos = list(self.root_dir.joinpath('train').iterdir())
            test_videos = list(self.root_dir.joinpath('test').iterdir())
            self.root_dir = self.root_dir.joinpath('train' if self.is_train else 'test')
        else:
            logging.info("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
        
        if self.is_train:
            self.videos = train_videos
            self.transform = build_transforms(self.augmentation_params)
        else:
            self.videos = test_videos
            self.transform = None
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            # id_sampling=True is not tested, because id_sampling in mgif/bair/fashion are False
            path = self.videos[idx]
            path = np.random.choice(list(path.glob('*.mp4')))
        else:
            path = self.videos[idx]
        video_name = path.name
        
        if self.is_train and path.is_dir():
            frames = sorted(path.iterdir(), key=lambda x: int(x.with_suffix('').name))
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [imread(frames[idx]) for idx in frame_idx]
        else:
            if self.create_frames_folder:
                video_array = read_video(path, frame_shape=self.frame_shape, saveto='folder')
                self.videos[idx] = path.with_suffix('')  # rename /xx/xx/xx.gif -> /xx/xx/xx
            else:
                video_array = read_video(path, frame_shape=self.frame_shape, saveto=None)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = [video_array[i] for i in frame_idx]
        
        # convert to 3-channel image
        if video_array[0].shape[-1] == 4:
            video_array = [i[..., :3] for i in video_array]
        elif video_array[0].shape[-1] == 1:
            video_array = [np.tile(i, (1, 1, 3)) for i in video_array]
        elif len(video_array[0].shape) == 2:
            video_array = [np.tile(i[..., np.newaxis], (1, 1, 3)) for i in video_array]
        out = {}
        
        if self.is_train:
            if self.transform is not None:
                t = self.transform(tuple(video_array))
                out['driving'] = t[0].transpose(2, 0, 1).astype(np.float32) / 255.0
                out['source'] = t[1].transpose(2, 0, 1).astype(np.float32) / 255.0
            else:
                source = np.array(video_array[0], dtype='float32') / 255.0  # shape is [H, W, C]
                driving = np.array(video_array[1], dtype='float32') / 255.0  # shape is [H, W, C]
                out['driving'] = driving.transpose(2, 0, 1)
                out['source'] = source.transpose(2, 0, 1)
            if self.time_flip and np.random.rand() < 0.5:
                buf = out['driving']
                out['driving'] = out['source']
                out['source'] = buf
        else:
            video = np.stack(video_array, axis=0).astype(np.float32) / 255.0
            out['video'] = video.transpose(3, 0, 1, 2)
            return out['video']
        out['name'] = video_name
        return out['driving'], out['source']
    
    def getSample(self, idx):
        return self.__getitem__(idx)


def build_transforms(augmentation_params):
    transform_list = []
    
    if 'flip_param' in augmentation_params:
        if 'horizontal_flip' in augmentation_params['flip_param']:
            transform_list.append(PairedRandomHorizontalFlip(keys=['image', 'image']))
    
    if 'crop_param' in augmentation_params or 'resize_param' in augmentation_params:
        transform_list.append(
            PairedRandomResizedCrop(size=augmentation_params['crop_param']['size'],
                                    scale=augmentation_params['resize_param']['ratio'],
                                    keys=['image', 'image'])
        )
    
    if 'jitter_param' in augmentation_params:
        transform_list.append(PairedColorJitter(
            **augmentation_params['jitter_param'],
            keys=['image', 'image']
        ))
    return T.Compose(transform_list)


# class FramesDataset(Dataset):
#     """
#     Dataset of videos, each video can be represented as:
#       - an image of concatenated frames
#       - '.mp4' or '.gif'
#       - folder with all frames
#     FramesDataset[i]: obtain sample from i-th video in self.videos
#     """
#
#     def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
#                  random_seed=0, pairs_list=None, augmentation_params=None, process_time=False, create_frames_folder=True):
#         self.root_dir = root_dir
#         self.videos = os.listdir(root_dir)
#         self.frame_shape = tuple(frame_shape)
#         self.pairs_list = pairs_list
#         self.id_sampling = id_sampling
#         self.process_time = process_time
#         self.create_frames_folder = create_frames_folder
#         if os.path.exists(os.path.join(root_dir, 'train')):
#             assert os.path.exists(os.path.join(root_dir, 'test'))
#             logging.info("Use predefined train-test split.")
#             if id_sampling:
#                 train_videos = {os.path.basename(video).split('#')[0] for video in
#                                 os.listdir(os.path.join(root_dir, 'train'))}
#                 train_videos = list(train_videos)
#             else:
#                 train_videos = os.listdir(os.path.join(root_dir, 'train'))
#             test_videos = os.listdir(os.path.join(root_dir, 'test'))
#             self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
#         else:
#             logging.info("Use random train-test split.")
#             train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
#
#         if is_train:
#             self.videos = train_videos
#         else:
#             self.videos = test_videos
#         self.buffed = [None]*len(self.videos)
#         self.is_train = is_train
#
#         if self.is_train:
#             self.transform = augmentation_params
#         else:
#             self.transform = None
#
#     def __len__(self):
#         return len(self.videos)
#
#     def colorize(self, image, hue):
#         """Hue disturbance
#         input range: [-1, 1]
#         """
#         res = rgb_to_hsv(image)
#         res[:, :, 0] = res[:, :, 0] + hue
#         res[:, :, 0][res[:, :, 0]<0] = res[:, :, 0][res[:, :, 0]<0] + 1
#         res[:, :, 0][res[:, :, 0]>1] = res[:, :, 0][res[:, :, 0]>1] - 1
#         res = hsv_to_rgb(res)
#         return res
#
#     def preload(self, idx):
#         """return the $idx$-th video
#         """
#         if self.is_train and self.id_sampling:
#             name = self.videos[idx]
#             path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
#         else:
#             name = self.videos[idx]
#             path = os.path.join(self.root_dir, name)
#         video_array = read_video(path, frame_shape=self.frame_shape)
#         return video_array
#
#     def __getitem__(self, idx):
#         if self.process_time:
#             a0 = time.process_time()
#         if self.is_train and self.id_sampling:
#             # id_sampling=True is not tested, because id_sampling in mgif/bair/fashion are False
#             name = self.videos[idx]
#             path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
#         else:
#             name = self.videos[idx]
#             path = os.path.join(self.root_dir, name)
#         video_name = os.path.basename(path)
#
#         if self.is_train and os.path.isdir(path):
#             frames = os.listdir(path)
#             num_frames = len(frames)
#             frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
#             video_array = [io.imread(os.path.join(path, frames[idx])) for idx in frame_idx]
#
#             # convert to 3-channel image
#             if video_array[0].shape[-1]==4:
#                 video_array = [i[..., :3] for i in video_array]
#             elif video_array[0].shape[-1]==1:
#                 video_array = [np.tile(i, (1, 1, 3)) for i in video_array]
#             elif len(video_array[0].shape)==2:
#                 video_array = [np.tile(i[..., np.newaxis], (1, 1, 3)) for i in video_array]
#         else:
#             if self.buffed[idx] is None:
#                 if self.create_frames_folder:
#                     video_array = read_video(path, frame_shape=self.frame_shape, saveto='folder')
#                     self.videos[idx] = name.split('.')[0]
#                 else:
#                     video_array = read_video(path, frame_shape=self.frame_shape, saveto=None)
#             else:
#                 video_array = self.buffed[idx]
#             num_frames = len(video_array)
#             frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
#                 num_frames)
#             video_array = [video_array[i] for i in frame_idx]
#         if self.process_time:
#             a1 = time.process_time()
#             print('Load T:%1.5f'%(a1-a0))
#         out = {}
#
#         # Dataset enhancement
#         if self.is_train:
#             source = np.array(video_array[0], dtype='float32')/255.0  # shape is [H, W, C]
#             driving = np.array(video_array[1], dtype='float32')/255.0 # shape is [H, W, C]
#             if len(driving.shape) == 2:
#                 driving = driving[..., np.newaxis]
#                 driving = np.tile(driving, (1, 1, 3))
#             if len(source.shape) == 2:
#                 source = source[..., np.newaxis]
#                 source = np.tile(source, (1, 1, 3))
#             if self.process_time:
#                 a11 = time.process_time()
#             # random_flip_left_right
#             if 'flip_param' in self.transform.keys() and self.transform['flip_param']['horizontal_flip']:
#                 if np.random.random() >= 0.5:
#                     driving = driving[:, ::-1, :]
#                     source = source[:, ::-1, :]
#             if self.process_time:
#                 a12 = time.process_time()
#                 print('A11-12 T:%1.5f'%(a12-a11))
#             # time_flip
#             if 'flip_param' in self.transform.keys() and self.transform['flip_param']['time_flip']:
#                 if np.random.random() >= 0.5:
#                     buf = driving
#                     driving = source
#                     source = buf
#             if self.process_time:
#                 a13 = time.process_time()
#                 print('A12-13 T:%1.5f'%(a13-a12))
#             # jitter_param 只写了hue
#             if 'jitter_param' in self.transform.keys():
#                 if 'hue' in self.transform['jitter_param'].keys():
#                     jitter_value = (np.random.random()*2-1)*self.transform['jitter_param']['hue']
#                     driving = self.colorize(driving, jitter_value)
#                     source = self.colorize(source, jitter_value)
#             if self.process_time:
#                 a14 = time.process_time()
#                 print('A13-14 T:%1.5f'%(a14-a13))
#             out['driving'] = driving.transpose((2, 0, 1))
#             out['source'] = source.transpose((2, 0, 1))
#         else:
#             video = np.stack(video_array, axis=0).astype(np.float32)/255.0
#             out['video'] = video.transpose((3, 0, 1, 2))
#             return out['video']
#         if self.process_time:
#             a2 = time.process_time()
#             print('Trans T:%1.5f'%(a2-a14))
#         out['name'] = video_name
#         return out['driving'], out['source']
#
#     def getSample(self, idx):
#         return self.__getitem__(idx)


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]  # 'driving'
        second = self.initial_dataset[pair[1]] # 'source':[channel, frame, h, w]
        return first, second[:, 0, :, :]


class PairedRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            return F.hflip(image)
        return image


class PairedRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4, 4. / 3),
                 interpolation='bilinear',
                 keys=None):
        super().__init__(size, scale, ratio, interpolation, keys=keys)
        self.param = None
    
    def _apply_image(self, img):
        if self.param is None:
            self.param = self._get_param(img)
        
        i, j, h, w = self.param
        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, self.interpolation)


class PairedColorJitter(T.BaseTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, keys=None):
        super().__init__(keys=keys)
        self.brightness = _check_input(brightness, 'brightness')
        self.contrast = _check_input(contrast, 'contrast')
        self.saturation = _check_input(saturation, 'saturation')
        self.hue = _check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
    
    def _get_params(self, input):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        
        if self.brightness is not None:
            brightness = random.uniform(self.brightness[0], self.brightness[1])
            f = lambda img: F.adjust_brightness(img, brightness)
            transforms.append(f)
        
        if self.contrast is not None:
            contrast = random.uniform(self.contrast[0], self.contrast[1])
            f = lambda img: F.adjust_contrast(img, contrast)
            transforms.append(f)
        
        if self.saturation is not None:
            saturation = random.uniform(self.saturation[0], self.saturation[1])
            f = lambda img: F.adjust_saturation(img, saturation)
            transforms.append(f)
        
        if self.hue is not None:
            hue = random.uniform(self.hue[0], self.hue[1])
            f = lambda img: F.adjust_hue(img, hue)
            transforms.append(f)
        
        random.shuffle(transforms)
        return transforms
    
    def _apply_image(self, img):
        for f in self.params:
            img = f(img)
        return img