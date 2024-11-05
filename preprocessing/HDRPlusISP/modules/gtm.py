# File: gtm
# Description: Global Tone Mapping (perform contrast enhancement with an S-shaped function)
# Created:  上午11:16 
# Author: Gongzhe Li
import math
import numpy as np
from .basic_module import BasicModule, register_dependent_modules
from .helpers import gammasRGB, minmax_norm

def enhanceContrast_(x, gain):
    # Apply an S-shaped contrast enhancement curve
    x -= gain * np.sin(2 * np.pi * x)
    # Clip the result
    return 0 if x < 0 else (1 if x > 1 else x)


class GTM(BasicModule):
    # perform contrast enhancement with an S-shaped function


    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps = 1e-6
        self.param = 0.1
        self.contrast_gain = cfg.gtm.contrast_gain  # [0,1.]


    def execute(self, data):
        # apply an S-shaped contrast enhancement curve
        image = data['rgb_image'].astype(np.float32)
        image = np.clip(image / (self.cfg.saturation_values.hdr), 0, 1.).astype(np.float32)
        Lw_ave = np.exp(np.mean(np.log(self.eps + image)))
        Lm = (self.param / Lw_ave) * image
        Lm_max = np.max(Lm)
        image = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
        # image = minmax_norm(image)
        # image = 3*np.power(image, 2) - 2*np.power(image, 3)
        image = np.clip(image, 0, 1.)
        data['rgb_image'] = image.astype(np.float32)
