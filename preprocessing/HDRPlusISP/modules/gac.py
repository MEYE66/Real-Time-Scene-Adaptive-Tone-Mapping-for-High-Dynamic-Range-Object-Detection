# coding: utf-8
# File: gac.py
# Description: Gamma Correction Compression
# Created: 2024-06-20  
# Author: Gongzhe Li
import numpy as np
from .basic_module import BasicModule
from .helpers import gammasRGB



class GAC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
    def execute(self, data):
        image = data['rgb_image'].astype(np.float32)
        image = gammasRGB(image, 'compress')
        image = np.clip(image, 0., 1.)
        data['rgb_image'] = image.astype(np.float32)


