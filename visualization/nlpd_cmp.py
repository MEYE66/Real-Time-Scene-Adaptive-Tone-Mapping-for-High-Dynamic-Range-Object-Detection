import os.path

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils import pltImg, pltHist, minmax_norm, load_hdr_raw, int_norm, tensor_to_numpy, numpy_to_tensor, to_npimage


BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24

def upsample(img, odd, filt):
    img = F.pad(img, (1, 1, 1, 1), mode='replicate')
    h = 2 * img.shape[2]
    w = 2 * img.shape[3]
    if img.is_cuda:
        o = torch.zeros([img.shape[0], img.shape[1], h, w], device=img.get_device())
    else:
        o = torch.zeros([img.shape[0], img.shape[1], h, w])
    o[:, :, 0:h:2, 0:w:2] = 4 * img
    o = F.conv2d(o, filt, padding=math.floor(filt.shape[2] / 2))
    o = o[:, :, 2:h - 2 - odd[0], 2:w - 2 - odd[1]]

    return o


def downsample(img, filt):
    pad = math.floor(filt.shape[2]/2)
    img = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    o = F.conv2d(img, filt)
    o = o[:, :, :img.shape[2]:2, :img.shape[3]:2]

    return o


def laplacian_pyramid_s(img, n_lev, filt):
    pyr = [0] * n_lev
    o = img

    for i in range(0, n_lev - 1):
        g = downsample(o, filt)
        h_odd = g.shape[2] * 2 - o.shape[2]
        w_odd = g.shape[3] * 2 - o.shape[3]
        pyr[i] = o - upsample(g, [h_odd, w_odd], filt)
        o = g

    pyr[n_lev - 1] = o
    return pyr


def build_nlp(img, n_lev, params):  # 求得原图的拉普拉斯金字塔
    npyr = [0] * n_lev

    # print(params['gamma'])
    img = torch.pow(img, 1 / params['gamma'])
    pyr = laplacian_pyramid_s(img, n_lev, params['F1'])

    for i in range(0, n_lev - 1):
        pad = math.floor(params['filts'][0].shape[2] / 2)
        apyr = F.pad(torch.abs(pyr[i]), (pad, pad, pad, pad), mode='replicate')
        den = F.conv2d(apyr, params['filts'][0]) + params['sigmas'][0]
        npyr[i] = pyr[i] / den

    pad = math.floor(params['filts'][1].shape[2] / 2)
    apyr = F.pad(torch.abs(pyr[n_lev - 1]), (pad, pad, pad, pad), mode='replicate')
    den = F.conv2d(apyr, params['filts'][1]) + params['sigmas'][1]

    npyr[n_lev - 1] = pyr[n_lev - 1] / den

    return npyr


def build_params():

    params = dict()
    params['gamma'] = 2.60
    params['filts'] = dict()
    params['filts'][0] = torch.tensor([[0.0400, 0.0400, 0.0500, 0.0400, 0.0400],
                                            [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                            [0.0500, 0.0400, 0.0500, 0.0400, 0.0500],
                                            [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                            [0.0400, 0.0400, 0.0500, 0.0400, 0.0400]],
                                           dtype=torch.float)
    params['filts'][0] = params['filts'][0].unsqueeze(0).unsqueeze(0)

    params['filts'][1] = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0], [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]],
                                           dtype=torch.float)
    params['filts'][1] = params['filts'][1].unsqueeze(0).unsqueeze(0)

    params['sigmas'] = torch.tensor([0.1700, 4.8600], dtype=torch.float)
    params['F1'] = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                                 [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                 [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                                 [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                 [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                                dtype=torch.float)
    params['F1'] = params['F1'].unsqueeze(0).unsqueeze(0)

    return params



def calibrate(image, smax, smin):
    image = (smax - smin) * image + smin
    # image = minmax_norm(image)
    return image






def main():
    ### HDR data
    root_path = "/mnt/data1/RoD/RAW"
    # image_path = "night-12499.npy"
    image_path = "day-06050.npy"


    # root_path = "/home/gongzheli/workspace/DAT-main/TMO_CAN-master/results/dualcan/"
    # image_path = "day-06050.npy.png"


    # ### LDR data
    data_path = os.path.join(root_path, image_path)
    data = np.load(data_path).astype(np.float32)
    # data = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
    data = minmax_norm(data)
    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    data = calibrate(data, 1e6, 10)
    pltImg(data)

    filters = build_params()
    data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    print(data_tensor.shape, data_tensor.min(), data_tensor.max())
    pyr = build_nlp(data_tensor, 5, filters)


    for idx in range(len(pyr)):
        img = pyr[idx]
        img_np = tensor_to_numpy(img)
        # img_np = img_np ** (1/2.6)
        img_np = minmax_norm(img_np)
        img_np = np.clip(img_np*255, 0, 255).astype(np.uint8)
        pltImg(img_np)
        # flag = cv2.imwrite(f"./tmp_results/out_{image_path}_{idx}.png", img_np)
        # print(flag)
    # flag = cv2.imwrite(f"./tmp_results/out_{image_path}.png", to_npimage(data))

    return


if __name__ == '__main__':
    main()


