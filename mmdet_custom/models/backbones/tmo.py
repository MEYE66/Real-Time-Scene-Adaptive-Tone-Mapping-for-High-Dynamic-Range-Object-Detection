# coding: utf-8
# File: tmo.py
# Description: Numpy helpers for image processing
# Created: 2024-08-14
# Author: Gongzhe Li
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import kornia
import math
from torch.nn.modules.batchnorm import _BatchNorm



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
    pad = math.floor(filt.shape[2] / 2)
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


class nlpclass():
    def __init__(self):
        super(nlpclass, self).__init__()
        self.params = dict()
        self.params['gamma'] = 2.60
        self.params['filts'] = dict()
        self.params['filts'][0] = torch.tensor([[0.0400, 0.0400, 0.0500, 0.0400, 0.0400],
                                                [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                                [0.0500, 0.0400, 0.0500, 0.0400, 0.0500],
                                                [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                                [0.0400, 0.0400, 0.0500, 0.0400, 0.0400]],
                                               dtype=torch.float)
        self.params['filts'][0] = self.params['filts'][0].unsqueeze(0).unsqueeze(0)

        self.params['filts'][1] = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0], [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0]],
                                               dtype=torch.float)
        self.params['filts'][1] = self.params['filts'][1].unsqueeze(0).unsqueeze(0)

        self.params['sigmas'] = torch.tensor([0.1700, 4.8600], dtype=torch.float)

        self.params['F1'] = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                                          [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                          [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                                          [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                          [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                                         dtype=torch.float)
        self.params['F1'] = self.params['F1'].unsqueeze(0).unsqueeze(0)

        self.exp_s = 2.00
        self.exp_f = 0.60

    def nlp(self, h_img, n_lev=None):
        if n_lev is None:
            # n_lev = math.floor(math.log(min(h_img.shape[2:]), 2)) - 2  # 求得金字塔的层数
            n_lev = 5
            filts_0 = self.params['filts'][0]
            filts_1 = self.params['filts'][1]
            sigmas = self.params['sigmas']
            F1 = self.params['F1']

        if h_img.is_cuda:
            filts_0 = filts_0.cuda(h_img.get_device())
            filts_1 = filts_1.cuda(h_img.get_device())
            sigmas = sigmas.cuda(h_img.get_device())
            F1 = F1.cuda(h_img.get_device())

        filts_0 = filts_0.type_as(h_img)
        filts_1 = filts_1.type_as(h_img)
        sigmas = sigmas.type_as(h_img)
        F1 = F1.type_as(h_img)

        self.params['filts'][0] = filts_0
        self.params['filts'][1] = filts_1
        self.params['sigmas'] = sigmas
        self.params['F1'] = F1

        h_pyr = self.b_nlp(h_img, n_lev, self.params)

        return h_pyr

    def b_nlp(self, img, n_lev, params):  # 求得原图的拉普拉斯金字塔

        npyr = [0] * n_lev

        img = torch.pow(img, 1 / params['gamma'])
        # img = torch.log(img)
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
        self.w_0 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001, affine=False)

    def forward(self, x):
        return self.w_0 * self.bn(x)


class lrelu(nn.Module):
    def __init__(self):
        super(lrelu, self).__init__()

    def forward(self, x):
        return torch.max(x * 0.2, x)


def build_net(norm=AdaptiveNorm, layer=5, width=32):
    layers = [
        nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
    ]
    for l in range(1, layer):
        layers += [nn.Conv2d(width, width, kernel_size=3, stride=1, padding=2 ** l, dilation=2 ** l, bias=False),
                   norm(width),
                   lrelu(),
                   ]
    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
        nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
    ]

    net = nn.Sequential(*layers)
    net.apply(weights_init)
    return net


def build_net_conv(norm=AdaptiveNorm, layer=5, width=32):
    layers = [
        nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
    ]
    for l in range(1, layer):
        layers += [nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                   norm(width),
                   lrelu(),
                   ]
    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
        nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
    ]

    net = nn.Sequential(*layers)
    net.apply(weights_init)
    return net


def build_fast_net(norm=AdaptiveNorm, layer=4, width=32):
    layers = [
        nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        # norm(width),
        lrelu(),
    ]
    for l in range(1, layer):
        layers += [nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False),
                   # norm(width),
                   lrelu(),
                   ]
    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        # norm(width),
        lrelu(),
        nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
    ]
    net = nn.Sequential(*layers)
    net.apply(weights_init)
    return net


class logCANTMO_base(nn.Module):
    def __init__(self, layer=4, num_feat=32):
        super(logCANTMO_base, self).__init__()
        self.conv_net = build_net(layer=layer)
        self.screen_max, self.screen_min = 300, 5
        self.scene_max, self.scene_min = 1e6, 1e3
        self.image_adaptive_local = nn.Sequential(
            nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_feat, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def ada_pre_calibration(self, image_rgb, image_val):
        image_val = (image_val - image_val.min()) / (image_val.max() - image_val.min())  # 0-1 minmax_norm
        image_down = F.interpolate(image_rgb, (256, 256), mode='bilinear', align_corners=True)
        H, W = image_rgb.shape[-2:]
        param_local = self.image_adaptive_local(image_down)
        param_local = F.interpolate(param_local, (H, W), mode='bilinear', align_corners=True)
        luminance_local = 10 ** (param_local * 4 + (1 - param_local) * 7)
        image_val = image_val * (luminance_local - 10) + 10
        return image_val



    def forward(self, image_rgb):
        image_hsv = kornia.color.rgb_to_hsv(image_rgb)
        image_val = image_hsv[:, 2:3, :, :]
        image_val = self.ada_pre_calibration(image_rgb, image_val)
        image_val = torch.log10(image_val)
        result = self.conv_net(image_val)
        result = self.post_calibration(result)
        image_hsv[:, 2:3, :, :] = result
        result = kornia.color.hsv_to_rgb(image_hsv)
        return result




def default_init_weights(module_list, scale=1, bias_fill=0.1, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def thop_model_computation(model, input_data):
    macs, params = thop.profile(model, inputs=(input_data,))
    macs, params = thop.clever_format([macs, params], "%.3f")

    print('{:<30}  {:<8}'.format('Computational complexity Macs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def ptflops_model_computation(model, input_data):
    with torch.cuda.device(0):
        macs, params = ptflops.get_model_complexity_info(model, tuple(input_data.shape[1:]), as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def inference_latency_gpu(model, input_data):
    # device = torch.device("cuda:7")
    device = torch.device("cuda:9")
    model = model.to(device)
    dummy_input = input_data.to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    print(f"inference latency :{mean_syn:<4} ms.")


def inference_profiler_gpu(model, inputs):
    device = torch.device("cuda:9")
    # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    activities = [ProfilerActivity.CUDA, ]
    sort_by_keyword = 'cuda' + "_time_total"

    model = model.to(device)
    inputs = inputs.to(device)
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=5))







