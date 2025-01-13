# coding: utf-8
# File: luminance_distribution.py
# Description: Numpy helpers for image processing
# Created: 2024-09-13  
# Author: Gongzhe Li
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"



def vis_luminance_range():
    from matplotlib import colors as mcolors
    from matplotlib.ticker import PercentFormatter
    np.random.seed(10)
    # plt.figure(figsize=(12,8))

    # 定义亮度范围
    low = 4
    high = 7

    interval = 0.03
    # 生成数据
    # 较多的端点数据
    num_endpoint_samples = 500  # 4和7的样本数
    data_low = np.random.uniform(low, low + interval, num_endpoint_samples // 3)  # 4到4.5
    data_high = np.random.uniform(high - interval, high, num_endpoint_samples // 2)  # 6.5到7
    data_mid = np.random.uniform(5.5-0.03, 5.5+0.03, 50)  # 6.5到7
    # 中间均匀分布
    num_middle_samples = 4000  # 中间部分样本数
    data_middle = np.random.uniform(low + interval, high - interval, num_middle_samples)  # 5到6均匀分布

    # 合并数据
    data = np.concatenate([data_low, data_high, data_mid, data_middle])
    # data = np.concatenate([data_middle, data_mid])


    # 绘制亮度分布图

    n_bins = 128  # 设定直方图的箱数
    N, bins, patches = plt.hist(data, bins=n_bins, density=True, alpha=0.7)
    # plt.hist(data, bins=256, color='dimgray', alpha=0.8, density=True)


    # 计算每个箱子的颜色
    # fracs = N / N.max()  # 将计数标准化到0-1范围
    # norm = mcolors.Normalize(fracs.min(), fracs.max())  # 规范化颜色范围
    # 获取最小值和最大值
    min_value = bins[np.argmin(N)]
    max_value = bins[np.argmax(N)]

    # 设置颜色
    for i, thispatch in enumerate(patches):
        # 计算当前箱子的边界
        bin_center = 0.5 * (bins[i] + bins[i + 1])
        if i<10:
            thispatch.set_facecolor('darkblue')  # 最小值箱子为绿色
        elif (i>120):
            thispatch.set_facecolor('darkgreen')
        else:
            thispatch.set_facecolor('dimgray')



    # 设置图形属性
    # plt.title('Luminance Distribution with Varying Colors', fontsize=20)
    plt.xlabel('Scale Factor ($\mathregular{log_{10}}$)', fontsize=27)
    plt.ylabel('Density', fontsize=27)
    plt.xlim(3.9, 7.1)
    plt.xticks(np.arange(4, 7.5, 0.5), fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(axis='y', alpha=0.75)
    # # plt.title('Luminance Distribution', fontdict={'fontsize': 20})
    # plt.xlabel('Luminance Range', fontdict={'fontsize': 20})
    # plt.ylabel('Density', fontdict={'fontsize': 20})
    # plt.xlim(3.9, 7.1)
    # plt.xticks(np.arange(4, 7.5, 0.5))

    # plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
    # plt.yticks(np.arange(0, 1.0, 0.5))
    # plt.xaxis.set_ticks_position() # 设置x坐标刻度数字或名称的位置
    # ax.xaxis.set_ticks_position(np.arange(0, 1.0, 0.1)) # 设置y坐标刻度数字或名称的位置
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('./luminance_distribution.png', bbox_inches='tight')
    # 显示图形
    plt.show()



def gray_scale_range():

    levels = 28

    gray_scale = np.linspace(0, 1, levels).reshape(1, -1)

    # 创建一个新的图形
    plt.figure(figsize=(10, 2))

    # 使用imshow显示灰度条
    plt.imshow(gray_scale, aspect='auto', cmap='gray', extent=[0, levels, 0, 1])
    # 设置图形标题和标签
    # 显示灰度条
    # 设置图形标题和标签

    plt.yticks([])  # 隐藏y轴刻度
    plt.xticks([])
    # plt.xticks(np.arange(0, 1024, 100), labels=np.arange(0, 1024, 100))  # 设置x轴刻度
    # tick_positions = [0, 100, 1000, 10000, 100000, 1000000, 10000000, 1600000000]
    # tick_labels = ['0.000001', '0.0001', '0.01', '1', '100', '10,000', '1 Million', '100 Million', '1.6 Billion']
    # tick_positions = [50, 200, 300, 400, 500, 600,  800, 1050, 1180]

    # plt.xticks(tick_positions, labels=tick_labels)  # 设置x轴刻度
    # plt.xticks(np.arange(40, 1150, 120), labels=tick_labels)  # 设置x轴刻度

    # plt.xlim([0, 1200])

    # plt.savefig('lum_color.png', dpi=900, bbox_inches='tight', edgecolor='blue')
    # 显示灰度条
    # plt.savefig('LDR_color.png', bbox_inches='tight')
    plt.show()
    # 0.000001 0.0001 0.01 1 100 10000 1Million 100 Million 1.6 Billion




def gray_scale_range2():
    # 定义8位灰度条的长度
    num_levels = 128  # 每个8位灰度条的长度
    transition_length = 64  # 过渡区域的长度

    # 生成第一个8位灰度条（0到255）
    first_gray = np.linspace(0, 255, num_levels//2, dtype=np.uint8)

    # 生成第二个8位灰度条（255到0）
    second_gray = np.linspace(256, 512, num_levels//2, dtype=np.uint8)

    third_gray = np.linspace(512, 768, num_levels//2, dtype=np.uint8)
    
    forth_gray = np.linspace(768, 1020, num_levels//2, dtype=np.uint8)


    # 创建过渡区域
    # transition = np.linspace(255, 0, transition_length, dtype=np.uint8)
    # 将两个8位灰度条和过渡区域拼接
    full_gray_scale = np.concatenate((first_gray, second_gray, third_gray, forth_gray))
    full_gray_scale_16bit = full_gray_scale.astype(np.uint16)


    gray_image = full_gray_scale_16bit.reshape(1, -1)

    # 创建新的图形
    plt.figure(figsize=(12, 2))
    plt.imshow(gray_image, aspect='auto', cmap='gray')
    # # 设置图形标题和标签
    # plt.title('16-bit Grayscale Bar with Transition Between Two 8-bit Bars')
    # plt.xlabel('Gray Level (0-65535)')
    plt.yticks([])  # 隐藏y轴刻度
    plt.xticks([])
    # plt.xticks(np.linspace(0, len(full_gray_scale) - 1, 8), labels=np.linspace(0, 65535, 8, dtype=int))  # 每256个像素标记一个刻度
    # # 设置x轴范围
    # plt.xlim(-1, len(full_gray_scale))  # 留出边距
    plt.savefig('hdr_color.png', bbox_inches='tight')
    # 显示图形
    plt.show()




def mobius_strip(t, w):
    """ 生成莫比乌斯带的坐标 """
    x = (1 + 0.5 * w * np.cos(t / 2)) * np.cos(t)
    y = (1 + 0.5 * w * np.cos(t / 2)) * np.sin(t)
    z = 0.5 * w * np.sin(t / 2)
    return x, y, z



def npy_load(path):
    img = np.load(path, allow_pickle=True)
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('night-16402.png', img)
    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    gray_scale_range2()
    # vis_luminance_range()
    # day_path = "/home/gongzheli/Download/day-05107.npy"
    # night_path = "/home/gongzheli/Download/night-16402.npy"
    # npy_load(night_path)
