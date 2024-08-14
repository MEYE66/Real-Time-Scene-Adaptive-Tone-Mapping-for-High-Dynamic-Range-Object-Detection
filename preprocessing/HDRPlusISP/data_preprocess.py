# coding: utf-8
# File: data_preprocess.py
# Description: Numpy helpers for image processing
# Created: 2024-07-10  
# Author: Gongzhe Li
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.exposure

from retinex import MSRCR, automatedMSRCR, MSRCP

BIT8, BIT16, BIT24 = 2**8, 2**16, 2**24

def pltImg(img):
    plt.figure()
    plt.imshow(img)
    plt.show()



if __name__ == '__main__':

    tmo_config = {
    "sigma_list": [15, 80, 250],
    # "sigma_list": [50, 100, 400],
    "G"         : 5.0,
    "b"         : 25.0,
    "alpha"     : 125.0,
    "beta"      : 46.0,
    "low_clip"  : 0.01,
    "high_clip" : 0.95
    }


    # data = np.load("/mnt/data1/RoD/RAW/day-02470.npy")
    data = np.load("/mnt/data1/RoD/RAW/night-12560.npy")
    data = data ** (1/2.2)
    # data = (data - data.min())/(data.max() - data.min())
    data = np.clip(data *255, 0, 255).astype(np.uint8)

    # tone_mapper = MSRCR(data, tmo_config['sigma_list'],
    #     tmo_config['G'],
    #     tmo_config['b'],
    #     tmo_config['alpha'],
    #     tmo_config['beta'],
    #     tmo_config['low_clip'],
    #     tmo_config['high_clip'])
    # tone_mapper = automatedMSRCR(data, tmo_config['sigma_list'])
    tone_mapper = MSRCP(data, tmo_config['sigma_list'], tmo_config['low_clip'], tmo_config['high_clip'],)
    pltImg(tone_mapper)


    # pltImg(out)
    # pltImg(data)
    # out = skimage.exposure.equalize_hist(data)






