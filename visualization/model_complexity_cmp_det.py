import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


def main():

    # color template
    # 229 174 56  yellow
    # 253 79 48  r
    # 48 162 219

    # parameter cluster:

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 27
    ##### x-axis:   inference latency
    ##### y-axis:   map-detection performance
    ################# for gpu  time

    # <100k:  Ours, REconfigISP,SCI
    x = [64.32, 89, 25.8, 18.7, 8] # 51 # cf4a43
    y = [21.1, 42.1, 23.3, 54.6, 51.8]
    area = (50) * radius**2
    ax.scatter(x[:3], y[:3], s=area, alpha=1, marker='.', c='#F7BC99', linewidths=2.0) # represent params
    # ax.scatter(x, y, s=area, alpha=0.8, marker='.', c=, edgecolors='white', linewidths=2.0) # represent params
    plt.annotate('SCI22', (x[0]-9, y[0]-3.5), fontsize=notation_size)
    plt.annotate('ReconfigISP21', (x[1]-25,y[1]-3), fontsize=notation_size)
    plt.annotate('Zero-DCE++21', (x[2]-8, y[2]+2.3), fontsize=notation_size)
    # ax.scatter(x[4], y[4], s=area, alpha=1, marker='.', c='#F7BC99', linewidths=2.0) # represent params
    

    plt.annotate('Ours ', xy=(x[3]-1.5, y[3]-3.4),  fontsize=notation_size, weight='bold')
    plt.annotate('Ours (Lite)', xy=(x[4]-1.8, y[4]-3.6),  fontsize=notation_size,weight='bold')

    # plt.annotate('Ours', xy=(x[2]+5, y[2]-1),  fontsize=notation_size,)


    x = [18.7]
    y = [54.6]
    ax.scatter(x, y, s=(50) * radius**2, alpha=1, marker='.', c='#F7BC99', linewidths=2.5, linestyle='-', edgecolor='brown') # represent params
    x = [8.]
    y = [51.8]
    ax.scatter(x, y, s=(50) * radius**2, alpha=1, marker='.', c='#F7BC99', linewidths=2.5, linestyle='-', edgecolor='brown') # represent params




    ax.axvline(x=16.67, color='brown', linewidth=2, linestyle='--')
    # plt.annotate('Ours', xy=(x[2]-1.5, y[2]-2.4), xytext=(x[2]+4.8, y[2]-6.6), fontsize=notation_size,
    #              # arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
    #              )
    plt.annotate('Real Time', xy=(16.67, 15), xytext=(6.3, 20), fontsize=notation_size,color='brown',
                 arrowprops=dict(color='brown', shrink=0.1, width=2, headwidth=6, ),
                 )
    plt.annotate('(60FPS)', xy=(6.3, 18), fontsize=notation_size,color='brown',)


    # 100k-300k:RAODNet, IANet
    x = [22.4, 28.3, ]
    y = [33.8, 51.4,]
    area = (120) * radius**2
    ax.scatter(x, y, s=area, alpha=1, marker='.', c='#92A4B5', linewidths=2.0) # represent params
    plt.annotate('IANet22', (x[0]-4, y[0]-3.8), fontsize=notation_size)
    plt.annotate('RAODNet23', (x[1]-6, y[1]-4), fontsize=notation_size)



    # 1000k+: AnscombeNet
    x = [123]
    y = [27.8]
    area = (250) * radius**2
    ax.scatter(x, y, s=area, alpha=1, marker='.', c='#b9caa9', linewidths=2.0) # represent params
    plt.annotate('AnscombeNet21', (123-50, 27.87-4.3), fontsize=notation_size)


    ############## for cpu time
    # HDRPlus ISP, CLAHE, Manituk
    x = [2100, 270, 302]
    y = [49.5, 50.5, 48.5]
    area = (50) * radius**2
    ax.scatter(x, y, s=area, alpha=1, marker='.', c='#F7BC99',  linewidths=2.0)
    plt.annotate('HDR ISP', (x[0]-810, y[0]-3), fontsize=notation_size)
    plt.annotate('CLAHE94', (x[1]-45, y[1]+1.5), fontsize=notation_size)
    plt.annotate('Mantiuk08', (x[2]-60, y[2]-3), fontsize=notation_size)

    x = [18.7]
    y = [54.6]
    # ax.scatter(x[0], y[0], alpha=1.0, marker='*', c='r', s=500)
    # plt.scatter(x[0], y[0], s=10000,marker='.', linewidths=2.0, edgecolors='black', linestyle='--')

    x = [8]
    y = [51.8] # [53.2]
    # ax.scatter(x[0], y[0], alpha=1, marker='*', c='none',edgecolor='r', linestyle='-', s=500)


    plt.xscale('log', base=10)
    plt.xlim((15, 1000), auto=True)  # Adjust as needed to highlight differences
    plt.ylim((15, 57))



    plt.xlabel('Inference Latency(ms)', fontsize=35)
    plt.ylabel('mAP(%)', fontdict={'fontsize': 35})
    plt.title('mAP v.s. Parameters vs. Latency', fontsize=35)

    h = [
        plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
            [40, 60, 80], ['#F7BC99', '#92A4B5', '#b9caa9'], [1, 1., 1, ])
    ]
    ax.legend(
        labelspacing=0.1,
        handles=h,
        handletextpad=0.5,
        markerscale=1.0,
        fontsize=17,
        title='Parameters (K)',
        title_fontsize=20,
        labels=['<100k','100k-300k', '1M+'],
        scatteryoffsets=[0.0],
        loc='lower right',
        ncol=5,
        shadow=False,
        handleheight=3)

    # ax.text(0.0, 0., 'text', transform=ax.transAxes, fontsize=24, va='bottom', ha='left')
    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize('30')

    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize('30')


    plt.grid(True,linestyle='-.', linewidth=1)
    # ax.grid(linestyle='-.', linewidth=0.5)
    # ax.grid(b=True, linestyle='-.', linewidth=0.5)
    # plt.savefig('./fig_model.png', bbox_inches='tight')
    plt.savefig('fig_model.pdf', bbox_inches='tight')
    plt.show()


def minmax(img):
    return (img - img.min()) / (img.max() - img.min())

def pltImg(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def gtm(img, eps=1e-5, param=0.1):
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = np.clip(out, 0, 1).astype(np.float32)
    return out


def color_norm(image):
    image = image.copy().astype(np.float32)
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)



if __name__ == '__main__':

    main()
    exit(234)
    # feature_map = cv2.imread("/mnt/data1/hdr_video/validation/temp/263_.tiff", cv2.IMREAD_UNCHANGED)
    feature_map = cv2.imread("/home/gongzheli/workspace/HDRPreprocessor-Detection/day-01-244_0.0.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

    # feature_map = cv2.imread("../lum-day-05069.tiff.png", cv2.IMREAD_UNCHANGED)
    # feature_map = feature_map**(0.9)
    # feature_map = feature_map **(1/2.2)
    # feature_map = minmax(feature_map)
    mean_r = feature_map[:, :, 0].mean().astype(np.float32)
    mean_g = feature_map[:, :, 1].mean().astype(np.float32)
    mean_b = feature_map[:, :, 2].mean().astype(np.float32)
    feature_map[:, :, 0] *= mean_g / mean_r
    feature_map[:, :, 2] *= mean_g / mean_b
    feature_map = minmax(feature_map)
    pltImg(feature_map)

    # main()




