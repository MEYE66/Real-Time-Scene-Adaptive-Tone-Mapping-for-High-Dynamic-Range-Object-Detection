import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


def main():
    # parameter cluster:

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 27
    ##### x-axis:   inference latency
    ##### y-axis:   map-detection performance
    ################# for gpu  time

    # <100k:  Ours, REconfigISP,SCI
    x = [64.32, 89, 32.7]
    y = [21.1, 42.1, 54.6]
    area = (75) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0) # represent params
    plt.annotate('SCI', (64.32-10, 21.1+1.5), fontsize=notation_size)
    plt.annotate('ReconfigISP', (89-25,42.1+2), fontsize=notation_size)
    plt.annotate('Ours', (32.7-4, 54.6-3.2), fontsize=notation_size)



    # 100k-300k:RAODNet, IANet, ZeroDCE
    x = [10.4, 15.3, 29.8,]
    y = [33.8, 51.7, 23.3,]
    area = (120) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#FFD93D', edgecolors='white', linewidths=2.0) # represent params
    plt.annotate('IA-ISPNet', (10.5-1.9, 33.8+2), fontsize=notation_size)
    plt.annotate('RAOD-ISPNet', (15.3-5, 51.7-3.5), fontsize=notation_size)
    plt.annotate('Zero-DCE++', (29.8-9, 23.3+2), fontsize=notation_size)


    # 1000k+: AnscombeNet
    x = [123]
    y = [27.8]
    area = (250) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#95CD41', edgecolors='white', linewidths=2.0) # represent params
    plt.annotate('AnscombeNet', (123-40, 27.8+3), fontsize=notation_size)


    ############## for cpu time
    # HDRPlus ISP, CLAHE, Manituke
    x = [2100, 270, 302]
    y = [49.5, 50.5, 48.5]
    area = (50) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('HDR ISP', (2100-850, 49.5+1.7), fontsize=notation_size)
    plt.annotate('CLAHE', (270-45, 50.5+1.5), fontsize=notation_size)
    plt.annotate('Manituke', (302-80, 48.5-2.5), fontsize=notation_size)

    x = [32.7]
    y = [54.4]
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=500)


    plt.xlim((100, 1e4), auto=True)

    plt.xscale('log')
    plt.ylim((15, 55),  auto=True)
    plt.xlabel('Inference Latency(ms)', fontsize=35)
    plt.ylabel('mAP(%)', fontsize=35)
    plt.title('mAP v.s. Parameters vs. Latency', fontsize=35)

    h = [
        plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
            [30, 60, 80], ['#4D96FF', '#FFD93D', '#95CD41'], [0.8, 1.0, 0.6, ])
    ]
    ax.legend(
        labelspacing=0.1,
        handles=h,
        handletextpad=0.5,
        markerscale=1.0,
        fontsize=20,
        title='Parameters (K)',
        title_fontsize=20,
        labels=['<100k','100k-300k', '1M+'],
        scatteryoffsets=[0.0],
        loc='best',
        ncol=4,
        shadow=True,
        handleheight=3)
    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize('30')
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize('30')

    # ax.grid(b=True, linestyle='-.', linewidth=0.5)
    ax.grid( linestyle='-.', linewidth=0.5)
    # ax.grid(b=True, linestyle='-.', linewidth=0.5)
    plt.show()
    fig.savefig('model_complexity_cmp_ours.pdf', pad_inches=0, bbox_inches='tight', dpi=900)


if __name__ == '__main__':
    main()
