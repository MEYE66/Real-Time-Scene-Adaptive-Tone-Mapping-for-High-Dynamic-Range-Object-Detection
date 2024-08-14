# File: awb
# Description: Auto White Balance (Gray World White Balance)
# Created:  上午11:17 
# Author: Gongzhe Li

import numpy as np

from .basic_module import BasicModule
from .helpers import split_bayer, reconstruct_bayer


def gray_world_gains(sub_arrays):
    r, gr, gb, b = sub_arrays
    r_gain = np.mean(r)
    gr_gain = np.mean(gr)
    gb_gain = np.mean(gb)
    b_gain = np.mean(b)
    k_gain = (r_gain + gr_gain + gb_gain + b_gain) / 4
    kr_gain = k_gain / r_gain
    kgr_gain = k_gain / gr_gain
    kgb_gain = k_gain / gb_gain
    kb_gain = k_gain / b_gain
    return kr_gain, kgr_gain, kgb_gain, kb_gain


class AWB(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.raw_bit = self.cfg.hardware.rgb_bit_depth - 1

    def execute(self, data):
        bayer = data['bayer'].astype(np.uint64)
        sub_arrays = split_bayer(bayer, self.cfg.hardware.bayer_pattern)
        gains = gray_world_gains(sub_arrays)

        wb_sub_arrays = []
        for sub_array, gain in zip(sub_arrays, gains):
            wb_sub_arrays.append(
                gain * sub_array,
                # np.right_shift((gain * sub_array), 10) # TODO: 10 -> self.raw_bit
            )
        wb_bayer = reconstruct_bayer(wb_sub_arrays, self.cfg.hardware.bayer_pattern)
        wb_bayer = np.clip(wb_bayer, 0, self.cfg.saturation_values.hdr)
        data['bayer'] = wb_bayer.astype(np.uint32)