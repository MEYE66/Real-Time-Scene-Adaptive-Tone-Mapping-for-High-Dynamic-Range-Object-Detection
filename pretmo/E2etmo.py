import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import random
import kornia

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:

        init.xavier_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.uniform_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]),requires_grad=True)

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001, affine=False, track_running_stats=True)


    def forward(self, x):
        # print(f"bn layer   mean:{torch.mean(self.bn.running_mean)} var:{torch.mean(self.bn.running_var)}   #########")
        return self.w_0 * self.bn(x)


class lrelu(nn.Module):
    def __init__(self):
        super(lrelu, self).__init__()
    def forward(self, x):
        return torch.max(x*0.2, x)



def build_net(norm=AdaptiveNorm, layer=5, width=32):
    layers = [
        nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False), # True
        norm(width),
        lrelu(),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(width),
                   lrelu(),
                   ]

    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
        nn.Conv2d(width,  1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
    ]

    net = nn.Sequential(*layers)
    net.apply(weights_init)
    return net



class ResTMO(nn.Module):
    """Tone Mapper optimized under NLPD Loss
    NLPD Loss optimize (scene) and (screen) luminance
    pre_calibration:  scene calibration
    post_calibration: screen calibration
    """
    def __init__(self, layer=4):
        super(ResTMO, self).__init__()
        self.d_max = 300
        self.d_min = 5
        
        self.conv_net = build_net(layer=layer)
        
    
    def pre_calibration(self, image_val):
        if self.training:
            s_min = 5.0
            s_max = random.choice([1e4,1e5,1e6,1e7]) #1e7 1e8  1e9
        else:
            s_min = 5.0
            s_max = 1e6
        image_val = (s_max - s_min) * image_val + s_min
        return image_val
    
    def post_calibration(self, image_val):
        image_val = torch.sigmoid(image_val)
        image_val = (self.d_max - self.d_min) * image_val + self.d_min
        return image_val 
    
    def repro_color(self, image_val, image_hsv):
        image_val = (image_val - image_val.min())/(image_val.max()-image_val.min())
        image_hsv[:,2:3, :, :] = image_val
        image_rgb = kornia.color.hsv_to_rgb(image_hsv)
        return image_rgb
    
    def forward(self, image_val, image_hsv=None):
        """
        """
        # image_val = torch.log10(image_val)
        # image_val = self.pre_calibration(image_val)
        image_val = self.conv_net(image_val)
        image_val = self.post_calibration(image_val)
        
        if image_hsv:
            image_rgb = self.repro_color(image_val, image_hsv)
            return image_rgb
        
        return image_val
    





class logCANTMO(nn.Module):
    """Adaptive Luminance Tone Mapper
    stage 1. training tone mapper use in NLPD Loss (dataset imp calibration)
    stage 2. trianing tone mapper with detector  (model imp ada-calibration )
    pre_calibration
    post_calibration:
    ada_luminance_predictor
    """
    def __init__(self, layer=4, num_feat=32):
        super(logCANTMO, self).__init__()
        self.conv_net = build_net(layer=layer)
        self.screen_max, self.screen_min = 300, 5
        self.scene_max, self.scene_min = 1e6, 1e3
        self.temperature_ = 10
        self.image_adaptive_local = nn.Sequential(
            nn.Conv2d(3, num_feat, kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_feat, 1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.Sigmoid(),
        )
    
    def pre_calibration(self, image_rgb, image_val):
        image_val = (image_val - image_val.min())/(image_val.max()-image_val.min()) # 0-1 minmax_norm
        s_max = torch.max(image_val) / (torch.exp(torch.mean(torch.log(1e-6+image_val))))
        s_min = 10.
        image_val = (s_max - s_min) * image_val + s_min
        return image_val
    
    def ada_pre_calibration(self, image_rgb, image_val):
        image_val = (image_val - image_val.min())/(image_val.max()-image_val.min()) # 0-1 minmax_norm
        image_down = F.interpolate(image_rgb, (256, 256), mode='bilinear', align_corners=True)
        H, W = image_rgb.shape[-2:]
        param_local = self.image_adaptive_local(image_down)
        param_local = torch.sigmoid(param_local)
        param_local = F.interpolate(param_local, (H, W), mode='bilinear', align_corners=True)
        luminance_local = torch.sin(0.5* torch.pi * param_local)*self.scene_max + torch.cos(0.5* torch.pi * param_local)*self.scene_min  # original tone-mapper architecture
        # luminance_local = 10**(param_local * 3 + (1-param_local)*6)
        image_val = image_val * (luminance_local - 10) + 10
        return image_val
    
    
    # def ada_pre_calibration(self, image_rgb, image_val):
    #     image_val = (image_val - image_val.min())/(image_val.max()-image_val.min()) # 0-1 minmax_norm
    #     image_down = F.interpolate(image_rgb, (256, 256), mode='bilinear', align_corners=True)
    #     H, W = image_rgb.shape[-2:]
    #     param_local = self.image_adaptive_local(image_down)
    #     param_local = torch.sigmoid(param_local * self.temperature_)
    #     param_local = F.interpolate(param_local, (H, W), mode='bilinear', align_corners=True)
    #     luminance_local = torch.sin(0.5* torch.pi * param_local)*self.scene_max + torch.cos(0.5* torch.pi * param_local)*self.scene_min
    #     image_val = image_val * (luminance_local - 10) + 10
    #     return image_val
    
    def post_calibration(self, image_rgb):
        result = torch.sigmoid(image_rgb) # [0, 1]
        result = (self.screen_max - self.screen_min) * result + self.screen_min
        result[result > self.screen_max] = self.screen_max
        result[result < self.screen_min] = self.screen_min
        result = (result - self.screen_min) / (self.screen_max - self.screen_min)
        return result


    def forward(self, image_rgb):
        image_hsv = kornia.color.rgb_to_hsv(image_rgb)
        image_val = image_hsv[:, 2:3, :, :]
        # image_val_nlp = self.pre_calibration(image_rgb, image_val)
        image_val = self.ada_pre_calibration(image_rgb, image_val)
        image_val = torch.log10(image_val)
        result = self.conv_net(image_val)
        result = self.post_calibration(result)
        image_hsv[:, 2:3, :, :] = result
        result = kornia.color.hsv_to_rgb(image_hsv)
        return result



class logCAN(nn.Module):
    def __init__(self, layer=4, num_feat=32):
        super(logCAN, self).__init__()
        self.conv_net = build_net(layer=layer)
        self.screen_max,self.screen_min = 300, 5
        self.scene_max, self.scene_min = 1e6, 1e3

    def ada_calibration(self, image_val):
        scene_max = (torch.max(image_val) / (torch.exp(torch.mean(torch.log(1e-6 + image_val)))))
        scene_min = torch.min(image_val)
        image_val = (scene_max - scene_min) * image_val + scene_min
        return image_val

    def pre_calibration(self, image_val):
        # fixed scene luminance calibration
        if self.training:
            scene_max = random.choice([1e4, 1e5, 1e6])
            scene_min = 5
        else:
            scene_max = 1e6
            scene_min = 5
        # easy adaptive luminance calibration
        # s_max = (torch.max(image_val) / (torch.exp(torch.mean(torch.log(1e-6 + image_val)))))
        # s_min = 0.
        image_val = (scene_max - scene_min) * image_val + scene_min
        return image_val

    def post_calibration(self, image_val):
        image_val = torch.sigmoid(image_val)
        image_val = (self.screen_max - self.screen_min) * image_val + self.screen_min
        return image_val

    def forward(self, image_val):
        # image_val = self.pre_calibration(image_val)
        image_val = self.ada_calibration(image_val)
        out = torch.log10(image_val)
        out = self.conv_net(out)
        out = self.post_calibration(out)
        return out





class E2ETMO(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self, layer=4):
        super(E2ETMO, self).__init__()
        self.cnn = build_net(layer=layer)
        self.luminance_cnn = build_net(layer=layer)
        self.d_max = 300
        self.d_min = 5
        


    def forward(self, x):

        nlev = x.__len__()
        # y = [0] * nlev
        z = [0] * nlev

        for nlp_index in range(nlev - 1):
            #    start = time.time()
            z[nlp_index] = self.cnn(x[nlp_index])
            # patch_res = z[nlp_index]
            # if((patch_res!=patch_res).any()):
            #     print(f"after cnn:{nlp_index}")
            # assert (torch.isnan(z[nlp_index].any())), print(f"after cnn:{nlp_index}")

        # print()
        z[nlev - 1] = self.luminance_cnn(x[nlev - 1])
        result = self.reconstract_nlp(z)
        result = self.Constraints(result)


        return result

    def init_model(self, path):
        self.cnn.load_state_dict(torch.load(path))

    def Constraints(self, result):
        result = torch.sigmoid(result) # [0, 1]
        result = (self.d_max - self.d_min) * result + self.d_min
        return result

    def reconstract_nlp(self, pyr):
        if pyr[0].is_cuda:
            filt = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]], dtype=torch.float,device=pyr[0].get_device()).unsqueeze(0).unsqueeze(0)
        else:
            filt = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                                 [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                 [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                                 [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                 [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]], dtype=torch.float).unsqueeze(0).unsqueeze(0)


        nlev = pyr.__len__()
        # for i in range(nlev):
        #     pyr[i] = pyr[i].unsqueeze(0)
        R = pyr[nlev - 1]
        for index in range(pyr.__len__() - 2, -1, -1):
            h_odd = R.shape[2] * 2 - pyr[index].shape[2]
            w_odd = R.shape[3] * 2 - pyr[index].shape[3]
            R = pyr[index] + self.upsample(R, [h_odd, w_odd], filt)
        return R

    def upsample(self, img, odd, filt):
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


    def zero_padding(self,nlp_list):
        length = len(nlp_list)
        z = [0]*length
        w = nlp_list[0].size(2)
        h = nlp_list[0].size(3)
        for i in range(length-2,0,-1):
            z[i] = F.pad(nlp_list[i],(math.floor((h - nlp_list[i].size(3))/2),math.ceil((h - nlp_list[i].size(3))/2),math.floor((w - nlp_list[i].size(2))/2),math.floor((w - nlp_list[i].size(2))/2)),mode='constant',value=0)
        k = torch.cat([nlp_list[0],z[1],z[2],z[3]],dim=0)
        return k


if __name__ == '__main__':
    # img_val = torch.randn(1,1,224,224)
    # img_rgb = torch.randn(1,3,224,224)
    # model =ResTMO()
    # out = model(x)
    # print(out.shape)
    
    # ckpt_path = "/home/ligongzhe/mmdetection/experiments/pretmo/E2ETMO-00019.pt"
    # state_dict = torch.load(ckpt_path)['state_dict']
    # model = E2ETMO(layer=4)
    # model.load_state_dict(state_dict)
    
    ckpt_path = "/home/ligongzhe/ckpt/ada_canlog-00019.pt"
    state_dict = torch.load(ckpt_path)['state_dict']
    print(state_dict)
    model = logCANTMO(layer=4)
    model.load_state_dict(state_dict)
    print(model)
    











