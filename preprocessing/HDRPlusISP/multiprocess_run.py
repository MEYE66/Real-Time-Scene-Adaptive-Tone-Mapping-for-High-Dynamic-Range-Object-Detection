# coding: utf-8
# File: multiprocess_run.py
# Description: Numpy helpers for image processing
# Created: 2024-06-20  
# Author: Gongzhe Li
import os.path
import cv2
import numpy as np
import  shutil
import multiprocessing
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

from pipeline import Pipeline
from utils.yacs import Config

BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24



def load_raw(path):
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw.reshape(1860, 2880, 3).astype(np.float32)
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16) # shape [1856, 2880, 1]
    # raw = minmax_norm(raw).astype(np.float32)
    raw = np.squeeze(raw).astype(np.int64)
    # return (raw / (BIT24 - 1)).astype(np.float32)  # norm to range [0, 1]
    return raw



def load_raw_video(path):
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw.reshape(1860, 2880, 3).astype(np.float32)
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16) # shape [1856, 2880, 1]
    # raw = minmax_norm(raw).astype(np.float32)
    raw = np.squeeze(raw).astype(np.int64)
    # return (raw / (BIT24 - 1)).astype(np.float32)  # norm to range [0, 1]
    return raw



def func(input_path, out_path, isp_func):
    raw_image = load_raw(input_path)
    # raw_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    out, _ = isp_func.execute(bayer=raw_image, save_intermediates=False, verbose=False)
    rgb_image = out['rgb_image']
    rgb_image = np.clip(rgb_image*255, 0., 255).astype(np.uint8)
    # rgb_image = cv2.resize(rgb_image, (1280, 1280), interpolation=cv2.INTER_LINEAR)
    save_path = os.path.join(out_path, os.path.basename(input_path))
    # print(save_path)
    cv2.imwrite(f"{save_path.replace('raw', 'tiff')}", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))




def main():
    # isp config
    config_path = "./configs/test.yaml"
    cfg = Config(config_path)
    isp_pipeline = Pipeline(cfg)

    # dataset path
    # in_path = "/home/lgz/data/RoD/RAWo/"
    # out_path = "/home/lgz/data/RoD/RGB"
    # in_path = "/home/ligongzhe/data/RAWo/"
    # out_path = "/home/ligongzhe/data/RAWtiff/"


    in_path = "/mnt/data1/RhoVision/raw/"
    out_path = "/mnt/data1/RhoVision/RGB/"

    shutil.rmtree(out_path, ignore_errors=True); os.makedirs(out_path)
    print(f'input  path: {in_path}')
    print(f'output path: {out_path}')

    lines = glob(os.path.join(in_path, '*.raw'))
    print(f'{len(lines)} raw images found')
    # threads = 1
    threads = multiprocessing.cpu_count() // 2
    print(f'{threads} threads')

    para = Parallel(n_jobs=threads, backend='threading')
    para(delayed(func)(filename, out_path, isp_pipeline) for filename in tqdm(lines))

    files_out = glob(os.path.join(out_path, '*.tiff'))
    print(f'output number: {len(files_out)}')

if __name__ == '__main__':
    main()
