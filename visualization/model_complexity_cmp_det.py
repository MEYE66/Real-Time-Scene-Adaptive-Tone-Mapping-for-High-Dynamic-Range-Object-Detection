def main():
    import matplotlib.pyplot as plt


    # parameter cluster:
    # <50: TMO, ReconfigISP, SCI
    # 5-100:Ours, Zero-DCE,
    # 100-300:IANet, RAODNet
    # 1M+: AnscombeISP

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 25
    '''0 - 10'''
    # HDRPlus ISP
    x = [2100]
    y = [49.5]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('HDR ISP', (2300 - 150, 49.5 + 1), fontsize=notation_size)

    '''<50k'''
    # CLAHE, Manituke, SCI, ReconfigISP
    x = [660, 402, 1831, 1290]
    y = [50.5, 48.5, 22.4, 42.1]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=1.0, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('CLAHE', (270 - 70, 50.5 + 1), fontsize=notation_size)
    plt.annotate('Manituke', (302+30 , 48.5 + 1), fontsize=notation_size)
    plt.annotate('SCI', (1831 - 70, 22.4 +1), fontsize=notation_size)
    plt.annotate('ReconfigISP', (1290 - 75, 42.1 + 1), fontsize=notation_size)


    '''50 - 100k'''
    # Zero-DCE, Ours
    x = [1105, 931]
    y = [24.8, 54.1]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=0.3, marker='.', c='#FFD93D', edgecolors='white', linewidths=2.0)
    plt.annotate('Zero-DCE', (1105 - 90, 24.8 +1), fontsize=notation_size)
    plt.annotate('Ours', (931 - 70, 51.1 + 0.10), fontsize=notation_size)

    '''100 - 300k'''
    # IANet,RAODNet
    x = [230, 335]
    y = [32.3, 46.1]
    area = (164) * radius**2
    ax.scatter(x, y, s=area, alpha=0.6, marker='.', c='#95CD41', edgecolors='white', linewidths=2.0)
    plt.annotate('IA-ISPNet', (150 + 0, 32.3 + 2), fontsize=notation_size)
    plt.annotate('RAOD-ISPNet', (335-60 , 46.1 - 3.5), fontsize=notation_size)

    '''Ours '''
    x = [931]
    y = [54.1]
    area = (50) * radius**2
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=300)
    # ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('Ours', (931 - 70, 51.1 + 0.10), fontsize=notation_size)


    # AnscomebeNet,  1M+
    x = [340]
    y = [26.6]
    area = 500 * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#EAE7C6', edgecolors='white', linewidths=2.0)
    plt.annotate('AnscomebeNet', (340 - 20, 26.6 + 4), fontsize=notation_size)

    # plt.xlim(0, 800)
    plt.xlim(100, 3000, auto=True)
    plt.ylim(20,  auto=True)
    plt.xlabel('Inference Latency(ms)', fontsize=35)
    plt.ylabel('mAP', fontsize=35)
    plt.title('mAP vs. Parameters vs. Latency', fontsize=35)

    h = [
        plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
            [30, 60, 80, 110], ['#4D96FF', '#FFD93D', '#95CD41', '#EAE7C6'], [0.8, 1.0, 0.6, 0.8,])
    ]
    ax.legend(
        labelspacing=0.1,
        handles=h,
        handletextpad=0.5,
        markerscale=1.0,
        fontsize=20,
        title='Parameters (K)',
        title_fontsize=20,
        labels=['<50k',  '50k-100k','100k-300k', '1M+'],
        scatteryoffsets=[0.0],
        loc='center right',
        ncol=5,
        shadow=True,
        handleheight=3)

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize('30')
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize('30')

    # ax.grid(b=True, linestyle='-.', linewidth=0.5)
    ax.grid( linestyle='-.', linewidth=0.5)
    plt.show()

    # fig.savefig('model_complexity_cmp_ours.pdf')


if __name__ == '__main__':
    main()
