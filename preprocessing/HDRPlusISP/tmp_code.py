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
    image = skimage.exposure.equalize_adapthist(image,clip_limit=0.0001)
    # image = skimage.exposure.equalize_adapthist(image, clip_limit=0.001)
    return image


def func(input_path, out_path):
    # raw_image = load_raw(input_path)
    raw_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    # raw_image = minmax_norm(raw_image)
    rgb_image = histeq(raw_image)
    
    rgb_image = np.clip(rgb_image*255, 0., 255).astype(np.uint8)
    save_path = os.path.join(out_path, os.path.basename(input_path))
    # print(save_path)
    cv2.imwrite(f"{save_path.replace('tiff', 'png')}", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))




def main():


    in_path = "/home/ligongzhe/data/RAWtiff/night-08503.tiff"
    # in_path = "/home/ligongzhe/data/RAWtiff/day-02603.tiff"

    out_path = "./"
    
    func(in_path, out_path)

   

if __name__ == '__main__':
    main()
