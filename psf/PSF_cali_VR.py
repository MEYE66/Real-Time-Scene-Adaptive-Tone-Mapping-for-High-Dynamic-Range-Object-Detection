import torch
from torch import nn
import time
from scipy import io
import numpy as np


# 空间分块、光谱分波段卷积
# 由于卷积的“去边”特性，对输入图像进行空间分块时，不同块之间有重叠，卷积后不同分块之间不再重叠，卷积输出图像保持和输入图像相同的空间尺寸
# 计算过程中“不”使用for循环，支持多样本并行批处理
# 使用pytorch内置函数实现，支持梯度反向传播
# 光谱波段可以分组简并，即连续的若干个光谱波段共用相同的卷积核，该功能同样支持并行批处理、支持梯度反向传播

# 卷积核的横纵尺寸可不同
# 横向纵向的空间分块数可不同，但须为奇数，否则程序报错
# 光谱波段简并分组后，输入图像原始的总波段数应能被单个小组中的波段数整除，否则程序报错
# 输入图像的空间分辨率可以不被分块数整除，对于不能整除的余数部分，不进行卷积运算，与输入时的值保持一致


# 定义paraConv功能
class spatio_spectral_paraConv(nn.Module):    # 通过在channel维度堆叠分块，利用pytorch的底层并行卷积算法，快速实现分块卷积计算

    def __init__(self, ks_h, ks_w, band, spectralStride, block_h, block_w, height, width, batchsize):    # 卷积核的空间尺寸只能是奇数，否则zero-pading会出问题
        super().__init__()

        if band % spectralStride != 0:
            raise RuntimeError('Number of spectral bands is not divisible by spectral stride')

        if block_h % 2 == 0:
            raise RuntimeError('Number of spatial blocks must be odd')

        if block_w % 2 == 0:
            raise RuntimeError('Number of spatial blocks must be odd')

        self.ks_h = ks_h    # 卷积核纵向尺寸
        self.ks_w = ks_w    # 卷积核横向尺寸。卷积核的横纵尺寸可不同
        self.band = band    # 输入图像的原始光谱波段数
        self.stride = spectralStride    # 光谱简并分组，分组后每组包含的波段数。图像的原始波段数应能被该数值整除，否则程序报错
        self.block_h = block_h      # 空间纵向分块的数量，须是奇数，否则程序报错
        self.block_w = block_w      # 空间横向分块的数量，须是奇数，否则程序报错
        self.height = height    # 输入图像的纵向分辨率
        self.width = width      # 输入图像的横向分辨率
        self.channel = band // spectralStride * block_h*block_w     # 并行堆叠的层数
        self.bs = batchsize     # 批处理的样本量

        self.h2 = nn.Conv2d(self.channel, self.channel, (ks_h, ks_w), groups=self.channel, bias=False)    # 卷积去边，故padding=0
        self.zeroPad_unfold = nn.ZeroPad2d((ks_w//2, ks_w//2, ks_h//2, ks_h//2))    # 左右上下
        self.zeroPad_fold = nn.ZeroPad2d((0, width%block_w, 0, height%block_h))

    def partition_and_unfold(self, x):    # block代表分块数，行列分块数可不同
        bs, l, m, n = x.shape
        if [bs, l, m, n] != [self.bs, self.band, self.height, self.width]:
            raise RuntimeError('Unmatched datacube size')

        x_reshape = x.reshape(bs, self.band//self.stride, self.stride, m,n).permute(0,2,1,3,4).reshape(bs*self.stride, self.band//self.stride, m,n)    # spectral degeneration
        remainder_h, remainder_w = m % self.block_h, n % self.block_w
        quotient_h, quotient_w = (m-remainder_h) // self.block_h, (n-remainder_w) // self.block_w
        x_reduced = x_reshape[:,:, :m-remainder_h, :n-remainder_w]    # 除不尽的分块余数丢弃，否则影响分块尺寸造成无法堆叠
        paucity_h, paucity_w = self.ks_h // 2, self.ks_w // 2
        x_padded = self.zeroPad_unfold(x_reduced)
        x_partitioned = x_padded.unfold(dimension=-2, size=quotient_h + 2*paucity_h, step=quotient_h).\
                unfold(dimension=-2, size=quotient_w + 2*paucity_w, step=quotient_w)    # 无论unfold的维度是哪个，都会在tensor的维度的最右边增加一维，故第二个unfold的dimension还是-2
        x_unfolded = x_partitioned.reshape(bs*self.stride, self.channel, quotient_h+2*paucity_h, quotient_w+2*paucity_w)    # unfold顺序：先分块后波段，其中分块内部排列顺序为先行后列
        return x_unfolded

    def fold_and_merge(self, x_unfolded, x):
        main_h, main_w = self.height - self.height % self.block_h, self.width - self.width % self.block_w  # 数据立方去掉分块除不尽的余数的尺寸
        block_size_h, block_size_w = main_h // self.block_h, main_w // self.block_w
        x_permuted = x_unfolded.reshape(self.bs, self.stride, self.band // self.stride, self.block_h, self.block_w, block_size_h, block_size_w)\
            .permute(0,1,2,3,5,4,6)    # 分块矩阵拼成大矩阵，不能让小分块独立reshape，否则会不按大矩阵的行列排列，应该使分块和分块索引交叉排列，故首先使用permute
        x_folded = x_permuted.reshape(self.bs, self.stride, self.band//self.stride, main_h, main_w)
        x_reshaped = x_folded.permute(0,2,1,3,4).reshape(self.bs, self.band, main_h, main_w)
        # x_merged = torch.cat((torch.cat((x_reshaped, code[main_h-self.height:, :self.width-main_w]), dim=0), code[:, main_w-self.width:]), dim=1)    # 该方法被下面语句替代
        x_merged = self.zeroPad_fold(x_reshaped)
        if (main_h-self.height) != 0:
            x_merged[:, :, main_h-self.height:, :] = x[:, :, main_h-self.height:, :]
        if (main_w-self.width) != 0:
            x_merged[:, :, :, main_w-self.width:] = x[:, :, :, main_w-self.width:]
        return x_merged

    def forward(self, x):
        x_unfolded = self.partition_and_unfold(x)
        x_h2 = self.h2(x_unfolded)
        y = self.fold_and_merge(x_h2, x)
        return y


def gaussianKernel1d(length, sigma):
    if sigma <= 0:
        # sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        raise RuntimeError('sigma <= 0')

    if length%2 == 0:
        center = length/2 - 0.5    # 由于python的索引从0开始，所以是减去0.5
    else:
        center = length // 2

    x = np.arange(length) - center
    kernel = np.exp(-(x ** 2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussianKernel2d(ks_h, ks_w, sigma):
    if sigma <= 0:
        # sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        raise RuntimeError('sigma <= 0')

    if ks_h%2 == 0:
        center_h = ks_h/2 - 0.5
    else:
        center_h = ks_h // 2
    if ks_w%2 == 0:
        center_w = ks_w/2 - 0.5
    else:
        center_w = ks_w // 2

    x, y = np.arange(ks_h) - center_h, np.arange(ks_w) - center_w
    xx, yy = np.meshgrid(x, y, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


# 参数定义
ks_h, ks_w, band, height, width = 5, 5, 9, 256, 256
block_h, block_w, spectralStride = 3, 3, 3
batchsize = 2

# GPU判定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda: {}'.format(torch.cuda.is_available()))

# paraConv初始化
paraConv = spatio_spectral_paraConv(ks_h, ks_w, band, spectralStride, block_h, block_w,
                                    height, width, batchsize).to(device)

# 卷积核参数赋值
dict_paraConv = paraConv.state_dict()
temp = [0 for k in range(paraConv.channel)]
for k in range(paraConv.channel):
    temp[k] = torch.from_numpy(gaussianKernel2d(ks_h, ks_w, 0.8)).unsqueeze(dim=0)
    # temp[k] = (torch.ones(ks_h_h2,ks_w_h2) * torch.randint(1, 3, (1,1))).unsqueeze(dim=0)    # 测试模型正确性时使用
kernelInit = torch.cat(temp, dim=0)
dict_paraConv['h2.weight'] = kernelInit.unsqueeze(dim=1).to(device)
paraConv.load_state_dict(dict_paraConv)


# test（无数据debug测试，正式使用时注释掉）

x = torch.rand(batchsize, band, height, width).to(device)
# h = torch.rand(band/spectralStride, ks_h, ks_w)
with torch.no_grad():
    y = paraConv(x)
torch.nn.init.kaiming_normal_(paraConv.h2.weight.data)
print(y)

# test end


# 数据载入

# 载入编码图案和相机拍摄图像
# x = ... 编码图案
# y = ... 相机拍摄图像

# 数据载入结束


# 优化参数设置
lr = 1e-1   # 学习率
reg_decay = 0   # weight decay in optimizer
reg_positive = 1e-3     # poisitive constraint in loss function
reg_weight = 1e-6       # energy constraint in PSF

lossFunction = nn.MSELoss().to(device)
errorFunction = nn.MSELoss().to(device)
optimizer = torch.optim.SGD(
    [{'params': paraConv.parameters()}], #{'params': model_h2.coe_spectral}],
        #{'params': model_h2.rectification}],
    lr=lr, weight_decay=reg_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5, patience=1e3, threshold=1e-8, min_lr=1e-5)

loss, epoch = 1, 0
tol = 8.05e-4      # 优化结束条件：损失函数阈值
while loss > tol:
    # while epoch < 2e4:
    paraConv.train()
    y_hat = paraConv(x)
    loss = lossFunction(y_hat, y)

    loss_positive = 0
    loss_weight = 0
    count = 0
    for para in paraConv.parameters():
        loss_positive = loss_positive + torch.mean(nn.functional.relu(-1 * para) ** 2)
        loss_weight = loss_weight + torch.mean((torch.norm(para, p=1, dim=[-1,-2])) ** 2)
        pass  # count += torch.prod(torch.tensor(para.shape))
    loss += reg_positive * loss_positive + reg_weight * loss_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1e2 == 0:
        print("-- -------------- {} -----------------".format(epoch + 1))
        print('Train loss: {}'.format(loss))

    scheduler.step(loss)
    epoch += 1

y_hat = paraConv(x)
error = errorFunction(y_hat, y)
print("Residual error: {}".format(error))

torch.save(paraConv, './PSF.pth')

pass