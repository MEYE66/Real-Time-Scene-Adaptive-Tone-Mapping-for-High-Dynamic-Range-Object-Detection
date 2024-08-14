import enum
import os

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '9')
import cv2
import gzip
import shutil
import argparse
import numpy as np
import multiprocessing

from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F

BIT8 = 2 ** 8
BIT16 = 2 ** 16
BIT24 = 2 ** 24


class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


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
    """Demosaicing of Bayer images using Malver-He-Cutler algorithm.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.

    Compared to Debayer2x2 this method does not use upsampling.
    Compared to Debayer3x3 the algorithm gives sharper edges and
    less chromatic effects.

    ## References
    Malvar, Henrique S., Li-wei He, and Ross Cutler.
    "High-quality linear interpolation for demosaicing of Bayer-patterned
    color images." 2004
    """

    def __init__(self, layout: Layout = Layout.RGGB):
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





def read_raw_24b(file_path, img_shape=(1, 1, 1856, 2880), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)

    return raw_data


def func(filename, debayer, out_path):
    im = read_raw_24b(filename)
    im = torch.from_numpy(im).cuda().float()

    with torch.no_grad():
        im = debayer(im).detach().cpu().numpy()

    im = im.squeeze(0).transpose(1, 2, 0)
    im = cv2.resize(im, (1280, 1280), interpolation=cv2.INTER_CUBIC)

    mean_r = im[:, :, 0].mean()
    mean_g = im[:, :, 1].mean()
    mean_b = im[:, :, 2].mean()
    im[:, :, 0] *= mean_g / mean_r
    im[:, :, 2] *= mean_g / mean_b
    img = np.clip(im, 0, BIT24 - 1).astype(np.int32)
    # img = (img - img.min())/(img.max()-img.min()).astype(np.float32)



    save_path = os.path.join(out_path, os.path.basename(filename))
    # np.save(save_path.replace('.raw', '.npy'), img)
    cv2.imwrite(save_path.replace('.raw', '.tiff'), img)
    
    
    # with gzip.GzipFile(save_path.replace('.raw', '.npy.gz'), 'w') as f:
    #     np.save(file=f, arr=im)


def main(args):
    in_path = os.path.realpath(args['path']).rstrip('/') + '/'
    out_path = in_path.replace('/RAWo/', '/RAWtiff/')
    assert os.path.isdir(in_path), f'Invalid path < {args["path"]} >!'
    assert in_path != out_path, f'in_path should NOT be the same as out_path!'

    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)
    print(f'input  path: {in_path}')
    print(f'output path: {out_path}')



    lines = glob(os.path.join(in_path, '*.raw'))
    print(f'{len(lines)} raw images found')
    # exit(234)

    debayer = Debayer3x3().cuda()
    # debayer = Debayer5x5().cuda()

    if args['threads'] in [0, 1]:
        print('Single thread')
        for fn in tqdm(lines):
            func(fn, debayer, out_path)

    else:
        if args['threads'] == -1:
            threads = multiprocessing.cpu_count() // 2
        else:
            threads = args['threads']
        print(f'{threads} threads')

        para = Parallel(n_jobs=threads, backend='threading')
        para(delayed(func)(filename, debayer, out_path) for filename in tqdm(lines))

    files_out = glob(os.path.join(out_path, '*.tiff'))
    print(f'output number: {len(files_out)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--path', type=str, default='/home/lgz/data/RoD/RAWo')
    parser.add_argument('-p', '--path', type=str, default='/home/ligongzhe/data/RAWo')
    parser.add_argument('-t', '--threads', type=int, default=-1)
    args = parser.parse_args()
    args = args.__dict__
    main(args)