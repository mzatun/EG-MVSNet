#!/usr/bin/env python

import numpy as np
import cv2
import re
import argparse
import matplotlib.pyplot as plt
import os
from struct import *

def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]

        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('depth_path')
    # args = parser.parse_args()
    # depth_path = args.depth_path
    save_path = '/file/py_files/Vit-MVSNet/MVS_Aerial_Building/test.png'
    depth_path = '/file/py_files/Vit-MVSNet/MVS_Aerial_Building/test/Depths/009_56/4/00000038.png'
    depth_image = cv2.imread(depth_path)
    size1 = len(depth_image)
    size2 = len(depth_image[1])
    e = np.ones((size1, size2, 3), dtype=np.float32)
    out_init_depth_image = depth_image / (e * 255) # 归一化 0-1之间
    out_init_depth_image = np.sum(out_init_depth_image, axis=2) # 3通道数据加和
    plt.imsave(save_path, out_init_depth_image, format='png')
