# coding: utf-8
# File: utils.py
# Description: Numpy helpers for image processing
# Created: 2024-09-02  
# Author: Gongzhe Li
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


BIT8, BIT16,BIT24 = 2 ** 8, 2 ** 16, 2 ** 24


def gamma(image, gamma=2.2):
    return image ** (1 / gamma)






def load_hdr_raw(file_path, img_shape=(1856, 2880), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    return raw_data


def minmax_norm(image):
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image



def to_npimage(image):
    image = minmax_norm(image)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def pltImg(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()



def pltHist2(img, log=False):
    plt.figure()
    img = img.flatten()
    plt.hist(img, bins="auto", density=False, histtype='bar', align='mid', facecolor='blue', log=log)
    plt.show()


def pltHist(img, bins=256):
    plt.figure()
    hist, bin_edges = np.histogram(img,  bins=bins)
    plt.bar(bin_edges[:-1], hist, width=0.1)
    plt.show()


def int_norm(image, max_val):
    image = (image * max_val).astype(np.int32)
    return image


def numpy_to_tensor(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)


def tensor_to_numpy(a: torch.Tensor):
    if a.shape[1] == 3:
        return a.squeeze(0).permute(1, 2, 0).numpy()
    else:
        return a.squeeze().numpy()

def tensor_to_npimage(a: torch.Tensor, unnormalize=True):
    a_np = tensor_to_numpy(a)
    if unnormalize:
        a_np = a_np * 255
    a_np = a_np.astype(np.uint8)
    return cv2.cvtColor(a_np, cv2.COLOR_RGB2BGR)


def npimage_to_torch(a, normalize=True, input_bgr=True):
    if input_bgr:
        a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
    a_t = numpy_to_torch(a)

    if normalize:
        a_t = a_t / 255.0

    return a_t
