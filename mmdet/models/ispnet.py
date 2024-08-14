# coding: utf-8
# File: ispnet.py
# Description: Numpy helpers for image processing
# Created: 2024-08-14  
# Author: Gongzhe Li
import torch
import torch.nn as nn
import torch.nn.functional as F


# from mmdet.models.backbones import isp_utils as N


def tanh_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return (torch.tanh(x) * 0.5 + 0.5) * (right - left) + left

        return activation

    return get_activation(l, r)


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, bias=True):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              padding=pad,
                              bias=bias)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


#  RAOD Net from  "Toward RAW Object Detection: A New Benchmark and A New Model"
class RAODNet(nn.Module):
    def __init__(self, num_in_ch=3, nf=32, tm_pts_num=8):
        super(RAODNet, self).__init__()
        self.tm_pts_num = tm_pts_num
        self.gamma_range = [7.0, 10.5]
        self.head1 = BaseConv(num_in_ch, nf, ksize=3, stride=2)
        self.body1 = BaseConv(nf, nf * 2, ksize=3, stride=2)
        self.body2 = BaseConv(nf * 2, nf * 4, ksize=3, stride=2)
        self.body3 = BaseConv(nf * 4, nf * 2, ksize=3)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.image_adaptive_gamma = nn.Sequential(
            nn.Linear(nf * 2, nf * 4, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(nf * 4, 3, bias=False)
        )
        self.head2 = BaseConv(num_in_ch, nf, ksize=3, stride=2)
        self.body_local1 = BaseConv(nf, nf * 2, ksize=3, stride=2)
        self.image_adaptive_local = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 2, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 2, tm_pts_num * 3 * 2, 3, stride=1, padding=1, bias=True),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(3, 16, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(16, 3, 1, stride=1, padding=0, bias=False),
        )

    def apply_gamma(self, img, params):
        params = tanh_range(self.gamma_range[0], self.gamma_range[1])(params)[..., None, None]
        img = (img) ** (1.0 / params)
        return img

    def apply_local(self, img, LTM):
        # ## img : [n, 3, h, w]
        # ## LTM : [n, 2 * 3 * tm_pts_num, h, w]
        n1, _, h1, w1 = img.shape
        n2, _, h2, w2 = LTM.shape
        assert n1 == n2 and h1 == h2 and w1 == w2, f'LTM has invalid shape < {LTM.shape} >!'

        LTM = tanh_range(0.1, 99.9)(LTM).reshape(n1, 6, self.tm_pts_num, h1, w1)
        LTM1, LTM2 = torch.split(LTM, 3, dim=1)
        LTM1 = LTM1 / torch.sum(LTM1, dim=2, keepdim=True)  # pieces
        LTM2 = LTM2 / torch.sum(LTM1 * LTM2, dim=2, keepdim=True)  # scales
        LTM1_ = [LTM1[:, :, :i].sum(2) for i in range(self.tm_pts_num + 1)]

        total_image = 0
        for i, point in enumerate(torch.split(LTM2, 1, dim=2)):
            total_image += torch.minimum(torch.clamp(img - LTM1_[i], 0, None), LTM1_[i + 1] - LTM1_[i]) * point.squeeze(
                2)
        return total_image

    def forward(self, img):
        img_down = F.interpolate(img, (256, 256), mode='bilinear', align_corners=True)
        h, w = img.shape[-2:]

        fea = self.head1(img_down)
        fea_s2 = self.body1(fea)
        fea_s4 = self.body2(fea_s2)
        fea_s8 = self.body3(fea_s4)
        fea_gamma = self.pooling(fea_s8)
        fea_gamma = fea_gamma.view(fea_gamma.shape[0], fea_gamma.shape[1])
        para_gamma = self.image_adaptive_gamma(fea_gamma)
        out_gamma = self.apply_gamma(img, para_gamma)

        fea_local = self.head2(img_down)
        fea_local = self.body_local1(fea_local)
        param_local = self.image_adaptive_local(fea_local)
        param_local = F.interpolate(param_local, (h, w), mode='bilinear', align_corners=True)
        out_local = self.apply_local(img, param_local)

        out = self.out_layer((out_local + out_gamma) / 2)
        return out


# IANet from adaptive yolo for adverse weather
class IANet(nn.Module):
    def __init__(self, num_in_ch=3, number_f=32):
        super(IANet, self).__init__()
        self.e_conv1 = nn.Conv2d(num_in_ch, 16, 3, 2, 1, bias=True)
        self.e_conv2 = nn.Conv2d(16, number_f, 3, 2, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 2, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 2, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f, number_f, 3, 2, 1, bias=True)
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 14)
        self.relu = nn.LeakyReLU(0.1)

    def WB_filter(self, img, W):
        # limit wb param range
        log_wb_range = torch.tensor(0.5)
        W = torch.exp(tanh_range(-log_wb_range, log_wb_range)(W))
        # normalize param with luminance
        W_n = 1.0 / (
                1e-5 + 0.27 * W[:, 0] + 0.67 * W[:, 1] +
                0.06 * W[:, 2])
        W = W * W_n.reshape(-1, 1)
        # print(W_n.shape)
        # print(W.shape)
        W = W.reshape(-1, 3, 1, 1)
        return img * W

    def gamma_filter(self, img, G):
        # limit gamma range
        log_gamma_range = torch.log(torch.tensor(3))
        G = torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(G))
        G = G.reshape(-1, 1, 1, 1)
        # apply to image
        img = torch.pow(torch.maximum(img, torch.tensor(0.0001)), G)
        return img

    def tone_filter(self, img, ltm):
        # limit ltm param
        ltm = tanh_range(0.5, 2)(ltm)
        ltm = ltm.reshape(-1, 1, 1, 8)
        # compute Tl
        T_l = torch.sum(ltm, dim=3) + 1e-30
        T_l = T_l.reshape(-1, 1, 1, 1)
        # compute tone-mapping
        total_img = img * 0
        # print(total_img.shape)
        # print(img.shape)
        # print(ltm.shape)
        for i in range(8):
            total_img += torch.clip((img - torch.tensor(i / 8)), 0, 1.0 / 8) * ltm[:, :, :, i].reshape(-1, 1, 1, 1)
        total_img = total_img * torch.tensor(8) / T_l
        return total_img

    def contrast_filter(self, img, alpha):
        # limit alpha
        shape = img.shape[-2:]
        alpha = torch.tanh(alpha)
        alpha = alpha.reshape(-1, 1, 1, 1)
        # create lum_img and limit its range
        lum_img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :, :] + 0.06 * img[:, 2, :, :]
        luminance = torch.minimum(torch.maximum(lum_img, torch.tensor(0.0)), torch.tensor(1.0))
        luminance = luminance.reshape(-1, 1, shape[0], shape[1])
        # compute EnLum
        contrast_lum = -torch.cos(torch.pi * luminance) * 0.5 + 0.5
        # compute En
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        # apply contrast filter
        img_out = alpha * contrast_image + (1 - alpha) * img
        return img_out

    def sharpen_filter(self, img, lamda):
        shape = img.shape[-2:]

        # limit lamda
        lamda = tanh_range(torch.tensor(0.0), torch.tensor(5))(lamda)
        lamda = lamda.reshape(-1, 1, 1, 1)
        # make gaussian kernel
        x = torch.arange(-12, 13, device=img.device, dtype=torch.float32)
        k = torch.exp(-0.5 * torch.square(x / torch.tensor(5)))
        k = k / torch.sum(k)
        kernel = torch.unsqueeze(k, 1) * k
        kernel = kernel.reshape(1, 1, 25, 25)
        # apply gaussian filter
        x1 = img[:, 0, :, :]
        x1 = x1.reshape(-1, 1, shape[0], shape[1])
        x2 = img[:, 1, :, :]
        x2 = x2.reshape(-1, 1, shape[0], shape[1])
        x3 = img[:, 2, :, :]
        x3 = x3.reshape(-1, 1, shape[0], shape[1])
        x1 = F.conv2d(x1, kernel, padding=12, stride=1)
        x2 = F.conv2d(x2, kernel, padding=12, stride=1)
        x3 = F.conv2d(x3, kernel, padding=12, stride=1)
        gaussian_img = torch.cat([x1, x2, x3], dim=1)
        # print(gaussian_img.shape)
        # apply sharpen filter
        out_img = (img - gaussian_img) * lamda + img
        return out_img

    def forward(self, img):
        low_img = F.interpolate(img, size=256, mode='bilinear', align_corners=True)
        out = self.relu(self.e_conv1(low_img))
        out = self.relu(self.e_conv2(out))
        out = self.relu(self.e_conv3(out))
        out = self.relu(self.e_conv4(out))
        out = self.relu(self.e_conv5(out))
        out = out.reshape(-1, 2048)
        # print(x5.shape)
        out = self.fc1(out)
        params = self.fc2(out)
        enhance_image = self.WB_filter(img, params[:, 0:3])
        enhance_image = self.gamma_filter(enhance_image, params[:, 3])
        enhance_image = self.tone_filter(enhance_image, params[:, 4:12])
        enhance_image = self.contrast_filter(enhance_image, params[:, 12])
        enhance_image = self.sharpen_filter(enhance_image, params[:, 13])
        return enhance_image


def model_computation(model, input_data):
    # print(input_data.shape)
    # print(tuple(input_data.shape[1:]))
    with torch.cuda.device(0):
        macs, params = ptflops.get_model_complexity_info(model, tuple(input_data.shape[1:]), as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def inference_latency_gpu(model, input_data):
    # device = torch.device("cuda:7")
    device = torch.device("cpu")
    model.to(device)
    dummy_input = input_data.to(device)
    # dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
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


if __name__ == '__main__':
    import ptflops
    import numpy as np

    x = torch.randn(1, 3, 1280, 1280)

    model = RAODNet(num_in_ch=3)
    # model = IANet(num_in_ch=3)
    # model = LiteISPNet(num_in_ch=1,)
    # out = model(x)
    # print(out.shape)

    inference_latency_gpu(model, x)


