# coding: utf-8
# File: multiprocess_run.py
# Description: Numpy helpers for image processing
# Created: 2024-06-20  
# Author: Gongzhe Li
import os.path
import cv2
import numpy as np
import torch
import multiprocessing
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

from pipeline import Pipeline
from utils.yacs import Config
from preprocessing.parse_raw import read_raw_24b, Debayer3x3

BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24

debayer = Debayer3x3().cuda()

# def load_raw_video(path, input_size=(1280, 720)):
#     # raw_data = np.fromfile(data_path, dtype=np.uint8)
#     img_shape = (frames, 720, 1280)
#     raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16  # 305971200
#     raw_data = raw_data.reshape(img_shape).astype(np.float32)
#     print(raw_data.shape)
#
#
#
#     raw = np.fromfile(path, dtype=np.uint8)
#     frames = (raw_data.shape)[0] / (input_size[0]*input_size[1]) // 3
#     frames = int(frames)
#     print(f"FPS:{frames}")
#
#
#     raw = raw.reshape(1280, 720, 3).astype(np.float32)
#     raw = np.split(raw, 3, axis=2)
#     raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16) # shape [1856, 2880, 1]
#     # raw = minmax_norm(raw).astype(np.float32)
#     raw = np.squeeze(raw).astype(np.int64)
#     # return (raw / (BIT24 - 1)).astype(np.float32)  # norm to range [0, 1]
#     return raw





def func(raw_image, index, out_path, isp_func):
    out, _ = isp_func.execute(bayer=raw_image, save_intermediates=False, verbose=False)
    rgb_image = out['rgb_image']
    rgb_image = np.clip(rgb_image*255, 0., 255).astype(np.uint8)
    # rgb_image = cv2.resize(rgb_image, (1280, 1280), interpolation=cv2.INTER_LINEAR)
    save_path = os.path.join(out_path, str(index)+'.png')
    # print(save_path)
    cv2.imwrite(f"{save_path}", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))



def raw_func(raw_image, index, out_path, isp_func):
    im = torch.from_numpy(raw_image).cuda().float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        im = debayer(im).detach().cpu().numpy()

    im = im.squeeze(0).transpose(1, 2, 0)
    # im = cv2.resize(im, (1280, 1280), interpolation=cv2.INTER_CUBIC)
    mean_r = im[:, :, 0].mean()
    mean_g = im[:, :, 1].mean()
    mean_b = im[:, :, 2].mean()
    im[:, :, 0] *= mean_g / mean_r
    im[:, :, 2] *= mean_g / mean_b
    img = np.clip(im, 0, BIT24 - 1).astype(np.int32)
    save_path = os.path.join(out_path, str(index)+'.tiff')
    # cv2.imwrite(f"{save_path}", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_path}",  img)
    # return


def main():
    # isp config
    config_path = "./configs/test.yaml"
    cfg = Config(config_path)
    isp_pipeline = Pipeline(cfg)

    root_path = "/mnt/data1/hdr_video/raw_data/9-28/"
    # for root in os.listdir(root_path):
        # print(root)
    # exit(234)
        # dataset path
    in_path = "/mnt/data1/hdr_video/raw_data/10-3/LUCID_TRI054S-C_222503282__20241003200357399_video4.raw"
    # in_path = f"/mnt/data1/hdr_video/raw_data/9-28/{root}"
    out_path = f"/mnt/data1/hdr_video/rgb/marval-{os.path.basename(in_path)}"  # raw path
    # shutil.rmtree(out_path, ignore_errors=True);
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(f'input  path: {in_path}')
    print(f'output path: {out_path}')
    input_size = (1860, 2880)
    raw_videos = np.fromfile(in_path, dtype=np.uint8)
    frames = (raw_videos.shape)[0] / (input_size[0]*input_size[1]) // 3
    frames = int(frames)

    img_shape = (frames, 1860, 2880)
    raw_data = raw_videos[0::3] + raw_videos[1::3] * BIT8 + raw_videos[2::3] * BIT16  # 305971200
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    print(f'{frames} raw images found')

    # threads = 1
    threads = multiprocessing.cpu_count() // 2
    print(f'{threads} threads')

    para = Parallel(n_jobs=threads, backend='threading')
    para(delayed(func)(raw_data[i, :, :], i, out_path, isp_pipeline) for i in tqdm(range(frames)))
    # para(delayed(raw_func)(raw_data[i, :, :], i, out_path, isp_pipeline) for i in tqdm(range(frames)))

    files_out = glob(os.path.join(out_path, '*.png'))
    print(f'output number: {len(files_out)}')


if __name__ == '__main__':
    main()


