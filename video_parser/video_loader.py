# coding: utf-8
# File: video_loader.py
# Description: Numpy helpers for image processing
# Created: 2024-08-30  
# Author: Gongzhe Li
import enum
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from video_parser.utils import split_bayer, reconstruct_bayer, gray_world_gains
from tqdm import tqdm

BIT8,BIT16,BIT24 = 2 ** 8, 2 ** 16, 2 ** 24
class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)



def masks_CFA_Bayer(shape, pattern='RGGB'):
    pattern = pattern.upper()
    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')


class OriginalDebayer3x3(nn.Module):
    def __init__(self):
        super(OriginalDebayer3x3, self).__init__()
        self.kernels = nn.Parameter(
            torch.tensor([
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
            ]).view(4, 1, 3, 3), requires_grad=False
        )
        self.index = nn.Parameter(
            torch.tensor([
                # dest channel r
                [4, 2],  # pixel is R,G1
                [3, 1],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [1, 3],  # pixel is R,G1
                [2, 4],  # pixel is G2,B
            ]).view(1, 3, 2, 2), requires_grad=False
        )

    def forward(self, x):
        B, C, H, W = x.shape
        xpad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        c = F.conv2d(xpad, self.kernels, stride=1)
        c = torch.cat((c, x), 1)  # Concat with input to give identity kernel Bx5xHxW

        rgb = torch.gather(
            c,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand in batch is faster than repeat
        )
        return rgb



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




class Debayer5x5(torch.nn.Module):
    def __init__(self, layout: Layout = Layout.BGGR):
        super(Debayer5x5, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    # G at R,B locations
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [-2,  4,  8,  4, -2], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R,B at G in R rows
                    # scaled by 16
                    [ 0,  0,  1,  0,  0], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [-2,  8, 10,  8, -2], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [ 0,  0,  1,  0,  0], # noqa

                    # R,B at G in B rows
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 1,  0, 10,  0,  1], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R at B and B at R
                    # scaled by 16
                    [ 0,  0, -3,  0,  0], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [-3,  0, 12,  0, -3], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [ 0,  0, -3,  0,  0], # noqa

                    # R at R, B at B, G at G
                    # identity kernel not shown
                ]
            ).view(4, 1, 5, 5).float() / 16.0,
            requires_grad=False,
        )
        # fmt: on

        self.index = nn.Parameter(
            torch.tensor([
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]).view(1, 3, 2, 2), requires_grad=False
        )
        self.index = torch.nn.Parameter(
            # Below, note that index 4 corresponds to identity kernel
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape
        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        planes = torch.cat(
            (planes, x), 1
        )  # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather(
            planes,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand for singleton batch dimension is faster
        )
        return torch.clamp(rgb, 0, 1)

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)





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
    # image = np.clip(image, 0, 1.)
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
    img = minmax_norm(img)
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = np.clip(out, 0, 1).astype(np.float32)
    return out




def rawLoad(raw, input_size=(1280, 1280), float_out=True):
    raw = np.squeeze(raw)
    # raw = (raw[0::3] + raw[1::3] * BIT8 + raw[2::3] * BIT16) # shape [1856, 2880, 1]
    raw = raw.reshape(*input_size).astype(np.int32)
    if float_out:
        # raw = minmax_norm(raw)
        raw = (raw / (BIT24-1))
    return raw

# debayer = Debayer3x3()
# debayer = OriginalDebayer3x3()
debayer = Debayer5x5()

def easyISP(raw_data, ):
    img = minmax_norm(raw_data)
    # img = raw_data / (BIT24 - 1)
    # img = raw_data
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    img = debayer(img)
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()
    out = gray_awb(img)
    return out
    tiff = out
    out = gtm(out)
    # out = np.clip((out*255).round(), 0, 255).astype(np.uint8)
    return out, tiff



def saveImage(image_path, image):
    image = np.clip(image*255, 0., 255.).astype(np.uint8)
    cv2.imwrite(image_path, image)


def pltImg(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()


def raw_test(file_path, img_shape=(1, 1, 1860, 2880), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    return raw_data




def raw_hdr_video():
    # data_path = "/mnt/data1/hdr_video/raw_data/LUCID_TRI054S-C_222503282__20240825155853743_video1.raw"  # 123
    # data_path = "/mnt/data1/hdr_video/raw_data/10-3/LUCID_TRI054S-C_222503282__20241003200857615_video7.raw" # 1480   school entry night
    # data_path = "/mnt/data1/hdr_video/raw_data/10-3/LUCID_TRI054S-C_222503282__20241003200357399_video4.raw"


    # night: f2.8-7-no-wb   night-01     LUCID_TRI054S-C_222503282__20240926221055626_video6.raw   night-02   LUCID_TRI054S-C_222503282__20240926221219797_video7.raw  night 04
    # day:f2.8-1-no-wb.raw day-01    f8-1-no-wb.raw -day-02     f8-3-no-wb.raw  -day-03
    data_path = "/mnt/data1/hdr_video/raw_data/10-23/f8-3-no-wb.raw"  #

    # data_path = "//mnt/data1/hdr_video/raw_data/hdr-videos/savedvideos/LUCID_TRI054S-C_222503282__20240926221219797_video7.raw"

    save_path = "/mnt/data1/hdr_video/validation/"
    folder = 'day-03'

    rgb_path = os.path.join(save_path, 'rgb', folder)
    tiff_path = os.path.join(save_path, 'tiff', folder)
    raw_path = os.path.join(save_path, 'raw_vis', folder)

    # data_path = "/home/gongzheli/data/hdr_video/raw_data/LUCID_TRI054S-C_222503282__20240913172338441_video1.raw" # 1482
    raw_data = np.fromfile(data_path, dtype=np.uint8)
    frames = (raw_data.shape)[0] / (2880*1860) // 3
    # frames = (raw_data.shape)[0] / (1280*1280) // 3


    frames = int(frames)
    print(f"FPS:{frames}")
    img_shape = (frames, 1860, 2880)
    # img_shape = (frames, 1280, 1280)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16  # 305971200
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    print(raw_data.shape)
    j = 0
    # frames = raw_data[200:, :, :]
    for i in tqdm(range(frames)):
        img_test = raw_data[i, :, :]
        # pltImg(img_test)tiff

        img_test = easyISP(img_test)
        out = minmax_norm(img_test)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{raw_path}/{i}.png", out)

        # # save rgb from isp
        # out, tiff = easyISP(img_test)
        # out = minmax_norm(out)
        # out = np.clip(out * 255, 0, 255).astype(np.uint8)
        # cv2.imwrite(f"{rgb_path}/{i}.png", out)
        #
        # # save tiff
        # tiff = minmax_norm(tiff)
        # tiff = np.clip(tiff * (BIT24-1), 0, (BIT24-1)).astype(np.int32)
        # cv2.imwrite(f"{tiff_path}/{i}_.tiff", tiff)

if __name__ == '__main__':
    raw_hdr_video()




