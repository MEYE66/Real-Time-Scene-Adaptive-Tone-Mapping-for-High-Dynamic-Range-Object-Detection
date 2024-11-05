import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import pltImg, pltHist, load_hdr_raw, int_norm, pltHist2, minmax_norm, to_npimage
import matplotlib as mpl


BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24



def gamma(image, gamma=1.0):
    return image**(1.0 / gamma)


if __name__ == '__main__':

    # root_path = "/mnt/data1/RoD/RAW"
    # image_path = "night-15663.npy"
    # # image_path = "day-06051.npy"
    # data_path = os.path.join(root_path, image_path)
    # data = np.load(data_path).astype(np.float32)
    # # data = data ** (1/1.5)
    # data = minmax_norm(data)


    # data_out = cv2.imread("/home/gongzheli/workspace/DAT-main/TMO_CAN-master/results/dualcan/day-06051.npy.png", cv2.IMREAD_UNCHANGED)
    # data_out = cv2.imread("/home/gongzheli/Download/night-15581.tiff", cv2.IMREAD_UNCHANGED)
    #
    #
    data_out = minmax_norm(data_out)
    # data_outbit24 = int_norm(data_out,  (BIT24 - 1))
    #
    #
    #
    # data_bit8 = int_norm(data_out, (BIT8 - 1))
    # data_bit16 = int_norm(data_out, (BIT16 - 1))
    data_bit24 = int_norm(data_out, (BIT24 - 1))
    #
    # data_8 = np.clip(data_bit24, 0, (BIT8 - 1)*3)
    # data_16 = np.clip(data_bit24, (BIT8 - 1)//2, (BIT16 - 1))
    # data_24 = np.clip(data_bit24, (BIT16 - 1)//10, (BIT24 - 1))
    data_all = np.clip(data_bit24, 0, (BIT24 - 1))
    #
    #
    # data_8 = gamma(data_8, 0.9)
    # data_16 = gamma(data_16, 5.2)
    # data_24 = gamma(data_24, 12.2)
    data_all = gamma(data_all, 10)
    #
    # data_8 = minmax_norm(data_8)
    # data_16 = minmax_norm(data_16)
    # data_24 = minmax_norm(data_24)
    data_all = minmax_norm(data_all)
    #
    # # pltImg(data_8)
    # # pltImg(data_16)
    # # pltImg(data_24)
    # cv2.imwrite("./tmp_results/8bit.png", to_npimage(data_8))
    # cv2.imwrite("./tmp_results/16bit.png", to_npimage(data_16))
    # cv2.imwrite("./tmp_results/24bit.png", to_npimage(data_24))
    # exit(234)
    #
    #
    # cv2.imwrite("./tmp_results/8bit.png", to_npimage(data_8))
    # cv2.imwrite("./tmp_results/16bit.png", to_npimage(data_16))
    # cv2.imwrite("./tmp_results/24bit.png", to_npimage(data_24))
    #
    # exit(234)

    # exit(234)

    # pltImg(data)
    # pltHist(np.log(data_all+1), bins=1024)
    # pltHist2(np.log(data_all+1))

    plt.figure(figsize=(6, 3))
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置matplotlib整体用Times New Roman
    # mpl.rcParams['font.weight'] = 'bold'  # 设置matplotlib整体用Times New Roman
    # mpl.rcParams['font.size'] = 26  # 设置matplotlib整体用Times New Roman
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4), useMathText=True)

    log_data = np.log(data_all+1)
    hist, bin_edges = np.histogram(log_data,  bins=256)
    plt.bar(bin_edges[:-1], hist, width=0.1,linewidth=1.5, color='dimgray', alpha=1)
    # plt.legend(loc='upper right')
    # plt.ylim((0, 12))
    plt.xlabel("Luminance in Log-scale")
    plt.ylabel("Pixel Number")

    plt.xlim(left=0, right=17.3)
    # plt.grid(visible=True, axis='both')
    # plt.savefig("./tmp_results/log_hist.png", bbox_inches='tight',pad_inches=0.05, dpi=900)

    plt.show()

    pass
