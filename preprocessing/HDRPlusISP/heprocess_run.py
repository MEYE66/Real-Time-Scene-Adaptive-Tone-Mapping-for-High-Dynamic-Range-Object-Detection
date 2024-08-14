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
import skimage.exposure

from retinex import MSRCP
import matplotlib.pyplot as plt

from pipeline import Pipeline
from utils.yacs import Config

BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24


def minmax_norm(image):
    image = (image - image.min())/(image.max() - image.min())
    return image.astype(np.float32)


def load_raw(path):
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw.reshape(1856, 2880, 3).astype(np.float32)
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16) # shape [1856, 2880, 1]
    # raw = minmax_norm(raw).astype(np.float32)
    raw = np.squeeze(raw).astype(np.int64)
    # return (raw / (BIT24 - 1)).astype(np.float32)  # norm to range [0, 1]
    return raw


def histeq(image):
    # TODO we use histogram equalization (skimage.exposure) to implement dynamic range compression    
    # for i in range(3):
    #     image[:, :, i] = skimage.exposure.equalize_hist(image[:, :, i])
    image = image ** (1/2.2)
    image = skimage.exposure.equalize_hist(image)
    return image


def retinex(image):
    image = image ** (1/2.2)
    image = (image - image.min())/(image.max() - image.min())
    image = np.clip(image *255, 0, 255).astype(np.uint8)
    image = MSRCP(image, [15, 80, 250], 0.01, 0.99)
    return image


def tonemapMantiuk(image, tonemapper):
    # cv2.createTonemapMantiuk()
    image = image ** (1/2.2)
    image = tonemapper.process(image)
    return image


def func(input_path, out_path, tonemapper):
    # raw_image = load_raw(input_path)
    raw_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    raw_image = minmax_norm(raw_image)
    # rgb_image = histeq(raw_image)
    # rgb_image = retinex(raw_image)
    rgb_image = tonemapMantiuk(raw_image, tonemapper)

    rgb_image = np.clip(rgb_image*255, 0., 255).astype(np.uint8)
    # rgb_image = np.clip(rgb_image, 0., 255).astype(np.uint8)
    save_path = os.path.join(out_path, os.path.basename(input_path))
    # print(save_path)
    cv2.imwrite(f"{save_path.replace('tiff', 'png')}", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


def main():
    # isp config
    # dataset path
    # in_path = "/home/lgz/data/RoD/RAWo/"
    # out_path = "/home/lgz/data/RoD/RGB"
    # tonemapper = cv2.createTonemapMantiuk()
    tonemapper = cv2.createTonemapReinhard(gamma=2.2)

    in_path = "/home/ligongzhe/data/RAWtiff/"
    out_path = "/home/ligongzhe/data/RGBrein/"

    shutil.rmtree(out_path, ignore_errors=True); os.makedirs(out_path)
    print(f'input  path: {in_path}')
    print(f'output path: {out_path}')

    lines = glob(os.path.join(in_path, '*.tiff'))
    print(f'{len(lines)} tiff images found')

    threads = multiprocessing.cpu_count() // 2
    print(f'{threads} threads')
    # exit(234)

    para = Parallel(n_jobs=threads, backend='threading')
    para(delayed(func)(filename, out_path, tonemapper) for filename in tqdm(lines))

    files_out = glob(os.path.join(out_path, '*.png'))
    print(f'output number: {len(files_out)}')

if __name__ == '__main__':
    main()
