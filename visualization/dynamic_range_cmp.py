import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import pltImg, pltHist, load_hdr_raw, int_norm, pltHist2, minmax_norm, to_npimage
import matplotlib as mpl


BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24



if __name__ == '__main__':

    root_path = "/mnt/data1/RoD/RAW"
    image_path = "night-15663.npy"
    # image_path = "day-06051.npy"
    data_path = os.path.join(root_path, image_path)
    data = np.load(data_path).astype(np.float32)
    # data = data ** (1/1.5)
    data = minmax_norm(data)


    data_out = cv2.imread("/home/gongzheli/workspace/DAT-main/TMO_CAN-master/results/dualcan/day-06051.npy.png", cv2.IMREAD_UNCHANGED)
    data_out = minmax_norm(data_out)
    data_outbit24 = int_norm(data_out,  (BIT24 - 1))



    data_bit8 = int_norm(data, (BIT8 - 1))
    data_bit16 = int_norm(data, (BIT16 - 1))
    data_bit24 = int_norm(data, (BIT24 - 1))

    data_8 = np.clip(data_bit24, 0, (BIT8 - 1))
    data_16 = np.clip(data_bit24, (BIT8 - 1), (BIT16 - 1))
    data_24 = np.clip(data_bit24, (BIT16 - 1), (BIT24 - 1))
    data_all = np.clip(data_bit24, 0, (BIT24 - 1))


    data_8 = minmax_norm(data_8)
    data_16 = minmax_norm(data_16)
    data_24 = minmax_norm(data_24) ** (1/2.9)

    # pltImg(data_8)
    # pltImg(data_16)
    # pltImg(data_24)
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

    plt.figure(figsize=(7, 4))
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置matplotlib整体用Times New Roman
    # mpl.rcParams['font.weight'] = 'bold'  # 设置matplotlib整体用Times New Roman
    # mpl.rcParams['font.size'] = 26  # 设置matplotlib整体用Times New Roman
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4), useMathText=True)

    log_data = np.log(data_all+1)
    hist, bin_edges = np.histogram(log_data,  bins=256)
    plt.bar(bin_edges[:-1], hist, width=0.1,linewidth=1.5, color='royalblue', alpha=1)
    # plt.legend(loc='upper right')
    # plt.ylim((0, 12))
    plt.xlabel("Luminance in Log-scale")
    plt.ylabel("Pixel Number")

    plt.xlim(left=0, right=17.3)
    # plt.grid(visible=True, axis='both')
    # plt.grid(axis='x')
    plt.savefig("./tmp_results/log_hist.png", bbox_inches='tight', dpi=300)
    plt.show()



    pass
