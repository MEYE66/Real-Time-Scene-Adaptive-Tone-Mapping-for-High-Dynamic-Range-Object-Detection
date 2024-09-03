# coding: utf-8
# File: video_loader.py
# Description: Numpy helpers for image processing
# Created: 2024-08-30  
# Author: Gongzhe Li
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from utils import split_bayer, reconstruct_bayer, gray_world_gains




BIT8,BIT16,BIT24 = 2 ** 8, 2 ** 16, 2 ** 24


def masks_CFA_Bayer(shape, pattern='RGGB'):
    pattern = pattern.upper()
    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')


class Debayer3x3(nn.Module):
    # This code is adjusted from the following url
    # https://github.com/cheind/pytorch-debayer/blob/master/debayer/modules.py
    def __init__(self):
        super(Debayer3x3, self).__init__()

        self.kernels = nn.Parameter(
            torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],

                [0, 0.25, 0],
                [0.25, 0, 0.25],
                [0, 0.25, 0],

                [0.25, 0, 0.25],
                [0, 0, 0],
                [0.25, 0, 0.25],

                [0, 0, 0],
                [0.5, 0, 0.5],
                [0, 0, 0],

                [0, 0.5, 0],
                [0, 0, 0],
                [0, 0.5, 0],
            ]).view(5, 1, 3, 3), requires_grad=False
        )
        self.index = nn.Parameter(
            torch.tensor([
                # dest channel r
                [0, 3],  # pixel is R,G1
                [4, 2],  # pixel is G2,B
                # dest channel g
                [1, 0],  # pixel is R,G1
                [0, 1],  # pixel is G2,B
                # dest channel b
                [2, 4],  # pixel is R,G1
                [3, 0],  # pixel is G2,B
            ]).view(1, 3, 2, 2), requires_grad=False
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        c = F.conv2d(x, self.kernels, stride=1)
        rgb = torch.gather(c, 1, self.index.repeat(B, 1, H // 2, W // 2))
        return rgb


def Malvar_demosaic(CFA, pattern='RGGB'):
    # linear demosiac use 3x3 kernel
    CFA = np.squeeze(np.array(CFA))
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)
    GR_GB = np.asarray(
        [[0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]]) / 8  # yapf: disable
    Rg_RB_Bg_BR = np.asarray(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, - 1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]]) / 8  # yapf: disable

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = np.asarray(
        [[0, 0, -1.5, 0, 0],
         [0, 2, 0, 2, 0],
         [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]]) / 8  # yapf: disable
    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    del G_m
    G = np.where(np.logical_or(R_m == 1, B_m == 1), convolve(CFA, GR_GB), G) # CFA:[H,W] GR_GB:[5,5] G:[2,4]
    RBg_RBBR = convolve(CFA, Rg_RB_Bg_BR)
    RBg_BRRB = convolve(CFA, Rg_BR_Bg_RB)
    RBgr_BBRR = convolve(CFA, Rb_BB_Br_RR)

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # Blue columns
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    del R_m, B_m

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c
    return cv2.merge([R, G, B])



def intAWB(bayer):
    sub_arrays = split_bayer(bayer, 'RGGB')
    gains = gray_world_gains(sub_arrays)

    wb_sub_arrays = []
    for sub_array, gain in zip(sub_arrays, gains):
        wb_sub_arrays.append(
            gain * sub_array,
            # np.right_shift((gain * sub_array), 10) # TODO: 10 -> self.raw_bit
        )
    wb_bayer = reconstruct_bayer(wb_sub_arrays, 'RGGB')
    wb_bayer = np.clip(wb_bayer, 0,None)
    return wb_bayer



def minmax_norm(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img.astype(np.float32)

def gray_awb(image):
    image = np.array(image)
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])
    image[:, :, 0] *= mean_g / mean_r
    image[:, :, 2] *= mean_g / mean_b
    image = np.clip(image, 0, 1.)
    return image


def white_awb(image):
    image = np.array(image)
    r_max = np.max(image[:, :, 0])
    g_max = np.max(image[:, :, 1])
    b_max = np.max(image[:, :, 2])
    r_gain = g_max / r_max
    b_gain = g_max / b_max
    image[:, :, 0] = r_gain * image[:, :, 0]
    image[:, :, 2] = b_gain * image[:, :, 2]

    return image



def gtm(img, eps=1e-6, param=0.5):
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = np.clip(out, 0, 1).astype(np.float32)
    return out


debayer = Debayer3x3()


def rawLoad(raw, input_size=(1280,1280), float_out=True):
    raw = np.squeeze(raw)
    raw = (raw[0::3] + raw[1::3] * BIT8 + raw[2::3] * BIT16) # shape [1856, 2880, 1]
    raw = raw.reshape(*input_size).astype(np.int32)
    if float_out:
        # raw = minmax_norm(raw)
        raw = (raw / (BIT24-1))
    return raw



def easyISP(raw_data):
    img = rawLoad(raw_data, input_size=(1, 1, 720, 1280), float_out=True)  #1280 × 720
    # img = rawLoad(raw_data, input_size=(720, 1280), float_out=True)  #1280 × 720

    # for .raw type
    # img = raw_data
    # img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    # for .avi type
    img = torch.from_numpy(img).float()
    #
    img = debayer(img)
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()
    # img = Malvar_demosaic(img)

    # img[:, :, 1] = img[:, :, 1]
    out = gray_awb(img)
    out = gtm(out)

    # out = minmax_norm(out)
    return out



def saveImage(image_path, image):
    image = np.clip(image*255, 0., 255.).astype(np.uint8)
    cv2.imwrite(image_path, image)


def pltImg(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def raw_test(file_path, img_shape=(1, 1, 1860, 2880), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    return raw_data


def video_main():
    video_path = "/mnt/data1/hdr_video/raw_data/indoor.avi"
    # video_name = os.path.basename(video_path).split('.')[0]
    save_path = f"/mnt/data1/hdr_video/save_data/test_video6/"
    # os.mkdir(save_path)
    vid_capture = cv2.VideoCapture(video_path)
    vid_capture.set(cv2.CAP_PROP_FORMAT, -1)
    print(f"Video FPS:{vid_capture.get(cv2.CAP_PROP_FPS)}, Video Time:{vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"Total Frames: {vid_capture.get(cv2.CAP_PROP_FPS)*vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)}")

    cnt = 0
    while True:
        ret, frame = vid_capture.read()
        # if not (cnt % 1):
        out = easyISP(frame)
        pltImg(out)
        # print(img.shape)
        # saveImage(save_path + str(cnt) + '.png', out)
        cnt += 1
        if cnt > 5:
            exit(234)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    vid_capture.release()
    print(f"frame num = {cnt}")
    return




def raw_hdr_video():
    data_path = "/mnt/data1/hdr_video/raw_data/LUCID_TRI054S-C_222503282__20240825155853743_video1.raw"  # 123
    raw_data = np.fromfile(data_path, dtype=np.uint8)

    # print(917913600/(720*1280))
    # print(113356800/(720*1280))
    # exit(234)
    img_shape = (123, 720, 1280)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16  # 305971200
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    print(raw_data.shape)

    for i in range(123):
        img_test = raw_data[i, :, :]
        # print(img_test.shape)
        # out = easyISP(img_test)
        out = minmax_norm(img_test)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f"/mnt/data1/hdr_video/save_data/video1raw/{i}.png", out)
        # pltImg(out)
        pltImg(img_test)
        if i > 20:
            break
        i += 1




if __name__ == '__main__':
    video_main()


    # data_path = "/mnt/data1/hdr_video/raw_data/LUCID_TRI054S-C_222503282__20240825155553368_video0.raw" # 332
