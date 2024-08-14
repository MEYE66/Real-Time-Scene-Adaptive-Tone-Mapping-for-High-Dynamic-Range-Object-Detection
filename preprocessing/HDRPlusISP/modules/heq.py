import numpy as np
import sys
import skimage.exposure

sys.path.append("../")

from .basic_module import BasicModule

class HEQ(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def execute(self, data):
        if self.params.mode.lower() == 'he':
            self.execute_he(data)
        elif self.params.mode.lower() == 'clahe':
            self.execute_clahe(data)
        elif self.params.mode.lower() == 'cs':
            self.execute_stretch(data)
        else:
            raise NotImplementedError



    def execute_clahe(self, data):
        # TODO we use histogram equalization (skimage.exposure) to implement dynamic range compression
        rgb_image = data['rgb_image'].astype(np.int32)
        out = np.clip(rgb_image / (self.cfg.saturation_values.hdr), 0, 1.).astype(np.float32)
        # for i in range(3):
        #     out[:, :, i] = skimage.exposure.equalize_hist(out[:, :, i])
            # out[:, :, i] = skimage.exposure.equalize_adapthist(out[:, :, i], clip_limit=0.05)
        out = skimage.exposure.equalize_adapthist(out, clip_limit=0.01)
        tmo_rgb_image = (np.clip(out, 0, 1.) * self.cfg.saturation_values.sdr).astype(np.uint8)  # *255
        data['rgb_image'] = tmo_rgb_image

    def execute_he(self, data):
        # TODO we use histogram equalization (skimage.exposure) to implement dynamic range compression
        # rgb_image = data['rgb_image'].astype(np.float32)
        image = data['rgb_image'].astype(np.uint64)
        image = np.clip(image / (self.cfg.saturation_values.hdr), 0, 1.).astype(np.float32)
    
        for i in range(3):
            image[:, :, i] = skimage.exposure.equalize_hist(image[:, :, i])
        tmo_rgb_image = (np.clip(image, 0, 1.) * self.cfg.saturation_values.sdr).astype(np.uint8)  # *255
        data['rgb_image'] = tmo_rgb_image


    def execute_stretch(self, data):
        # TODO we use histogram equalization (skimage.exposure) to implement dynamic range compression
        rgb_image = data['rgb_image'].astype(np.int32)
        out = np.clip(rgb_image / (self.cfg.saturation_values.hdr), 0, 1.).astype(np.float32)
        p2, p98 = np.percentile(out, (2, 98))
        out = skimage.exposure.rescale_intensity(out, in_range=(p2, p98))
        # for i in range(3):
        #     out[:, :, i] = skimage.exposure.rescale_intensity(out, in_range=(p2, p98))
        tmo_rgb_image = (np.clip(out, 0, 1.) * self.cfg.saturation_values.sdr).astype(np.uint8)  # *255
        data['rgb_image'] = tmo_rgb_image


