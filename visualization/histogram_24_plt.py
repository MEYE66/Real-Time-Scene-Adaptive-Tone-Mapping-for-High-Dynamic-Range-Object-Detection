import numpy as np
import shutil
import argparse
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt


BIT8  = 2 ** 8
BIT16 = 2 ** 16
BIT24 = 2 ** 24

def read_raw_24b(file_path, img_shape=(1856, 2880), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.int32)
    return raw_data

def draw_histogram(path):
    raw = read_raw_24b(path)
    r = raw[::2, ::2].flatten()
    g1 = raw[::2, 1::2].flatten()
    g2 = raw[1::2, ::2].flatten()
    b = raw[1::2, 1::2].flatten()

    hist_r_10bit, _ = np.histogram(r, bins=64, range=(0, 2**10-1))
    hist_r_16bit, _ = np.histogram(r, bins=128, range=(2**10, 2**16-1))
    hist_r_18bit, _ = np.histogram(r, bins=32, range=(2**16, 2**18-1))
    hist_r_24bit, _ = np.histogram(r, bins=32, range=(2**18, 2**24-1))
    hist_r = np.concatenate([hist_r_10bit, hist_r_16bit, hist_r_18bit, hist_r_24bit])
    
    hist_g_10bit, _ = np.histogram(g1, bins=64, range=(0, 2**10-1))
    hist_g_16bit, _ = np.histogram(g1, bins=128, range=(2**10, 2**16-1))
    hist_g_18bit, _ = np.histogram(g1, bins=32, range=(2**16, 2**18-1))
    hist_g_24bit, _ = np.histogram(g1, bins=32, range=(2**18, 2**24-1))
    hist_g = np.concatenate([hist_g_10bit, hist_g_16bit, hist_g_18bit, hist_g_24bit])

    hist_b_10bit, _ = np.histogram(b, bins=64, range=(0, 2**10-1))
    hist_b_16bit, _ = np.histogram(b, bins=128, range=(2**10, 2**16-1))
    hist_b_18bit, _ = np.histogram(b, bins=32, range=(2**16, 2**18-1))
    hist_b_24bit, _ = np.histogram(b, bins=32, range=(2**18, 2**24-1))
    hist_b = np.concatenate([hist_b_10bit, hist_b_16bit, hist_b_18bit, hist_b_24bit])
    # print("r: ",hist_r)
    # print("sum: ",sum(hist_r))

    hist = hist_r + hist_g + hist_b
    assert len(hist) == 256, f'len(hist) should be 256, but got {len(hist)}!'
    assert sum(hist) == (0.75) * (1856 * 2880), f'sum(hist) should be 0.75 * 1856 * 2880, but got {sum(hist)}!'
    # print(hist)
    return hist

def main(args):
    in_path  = os.path.realpath(args['path']).rstrip('/') + '/'
    out_path = in_path.replace('/raws/', '/histogram_24_plt/')
    assert os.path.isdir(in_path), f'Invalid path < {args["path"]} >!'
    assert in_path != out_path, f'in_path should NOT be the same as out_path!'

    shutil.rmtree(out_path, ignore_errors=True); os.makedirs(out_path)
    print(f'input  path: {in_path}')
    print(f'output path: {out_path}')

    lines = glob(os.path.join(in_path, '*.raw'))
    print(f'{len(lines)} raw images found')


    print('Single thread')
    draw_num = 0
    # lines.sort()
    for fn in tqdm(lines):
        if draw_num > 50:
            break
        im = draw_histogram(fn)
        save_path = os.path.join(out_path, os.path.basename(fn))
        save_path = save_path.replace('.raw', '.png')
        plt.bar(range(256), im)
        plt.title("Histogram")
        plt.xlabel("Value(2^10(64), 2^16(128), 2^18bit(32)), 2^24bit(32))")
        plt.ylabel("Counts")
        plt.savefig(save_path)
        plt.close()
        draw_num += 1
        # print(im)
        # print(im.shape)
        # exit()
    
    files_out = glob(os.path.join(out_path, '*.png'))
    print(f'output number: {len(files_out)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',    type=str)
    args = parser.parse_args()
    args = args.__dict__
    main(args)