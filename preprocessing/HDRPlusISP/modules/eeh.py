# File: eeh
# Description: perform shapening with unsharp making he mask is a linear combination of convolutions of the input image with 3 gaussian kernels of different sizes
# Created:  上午11:17 
# Author: Gongzhe Li

import numpy as np
import cv2
from .basic_module import BasicModule
from .helpers import pad, split_bayer, reconstruct_bayer, shift_array



def distL1_(x, y):
	# return y - x if y > x else x - y
    return np.abs(y-x)


def sharpenTriple_(x, b0, l0, th0, k0, b1, l1, th1, k1, b2, l2, th2, k2):
    # Compute the three sharpened values
    # r0 = x if l0 < th0 else x + k0 * (x - b0)
    # r1 = x if l1 < th1 else x + k1 * (x - b1)
    # r2 = x if l2 < th2 else x + k2 * (x - b2)
    r0 = np.where(l0 < th0, x, x + k0 * (x - b0))
    r1 = np.where(l1 < th1, x, x + k1 * (x - b1))
    r2 = np.where(l2 < th2, x, x + k2 * (x - b2))
    # Average them
    r = (r0 + r1 + r2) / 3.0
    # Clip the result
    return np.clip(r, 0, 1.)


class EEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        # performa sharpen with unsharp mask method
        # mask is a linear combination of convolutions of the input image
        # 	with 3 gaussian kernels of different sizes
        self.sigmas = cfg.eeh.sharpen_sigmas   # [1, 0.5, 0.5],
        self.amounts = cfg.eeh.sharpen_amounts  # [1,2,4]
        self.thresholds = cfg.eeh.sharpen_thresholds # [0.02, 0.04, 0.06]

    def execute(self, data):
        # apply an S-shaped contrast enhancement curve
        image = data['rgb_image'].astype(np.float32)

        # Compute all Gaussian blur
        blur0 = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=self.sigmas[0])
        blur1 = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=self.sigmas[1])
        blur2 = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=self.sigmas[2])

        # Compute all low contrast images
        low0 = distL1_(blur0, image)
        low1 = distL1_(blur1, image)
        low2 = distL1_(blur2, image)

        sharpImage = sharpenTriple_(image, blur0, low0, self.thresholds[0], self.amounts[0], blur1, low1, self.thresholds[1],
                                    self.amounts[1], blur2, low2, self.thresholds[2], self.amounts[2])
        sharpImage = np.clip(sharpImage, 0, 1)
        # tmo_rgb_image = (np.clip(sharpImage, 0, 1.)*self.cfg.saturation_values.sdr).astype(np.uint8) #*255
        data['rgb_image'] = sharpImage.astype(np.float32)

