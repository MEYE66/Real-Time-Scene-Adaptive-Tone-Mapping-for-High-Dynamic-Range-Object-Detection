import glob

import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    data = np.load("/home/lgz/data/RoD/RAW_debayer_fp32_1280x1280/day-05087.npy")
    print(data.shape, data.min(), data.max())
    plt.figure()
    plt.imshow(data)
    plt.show()