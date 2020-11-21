import os

import logging
from tqdm import trange
import imageio
from scipy.spatial import ConvexHull
from frames_dataset import PairedDataset
from train import load_ckpt
import numpy as np

# from sync_batchnorm import DataParallelWithCallback


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
            jacobian_diff = np.matmul(kp_driving['jacobian'].numpy(), np.linalg.inv(kp_driving_initial['jacobian'].numpy()))
            kp_new['jacobian'] = dygraph.to_variable(np.matmul(jacobian_diff, kp_source['jacobian'].numpy()))

    return kp_new
