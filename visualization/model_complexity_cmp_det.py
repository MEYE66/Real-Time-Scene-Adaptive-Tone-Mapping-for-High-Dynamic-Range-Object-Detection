def main():
    import matplotlib.pyplot as plt
    # parameter cluster:

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 25

    ##### x-axis:   inference latency
    ##### y-axis:   map-detection performance

    ################# for gpu  time
    #  IANet, RAODNet,ZeroDCE, SCI, DiffISP, AnscombeNet  Ours
    x = [10.4, 15.3, 29.8, 64.32, 89, 123, 32.7]
    y = [32.2, 46.1, 24.8, 22.4, 26.8, 42.1, 54.9]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0) # represent params
    plt.annotate('IA-ISPNet(GPU)', (10.5 + 0, 32.3 + 2), fontsize=notation_size)
    plt.annotate('RAOD-ISPNet(GPU)', (15.3-60, 46.1 - 3.5), fontsize=notation_size)
    plt.annotate('Zero-DCE++(GPU)', (29.8-60, 24.8 - 3.5), fontsize=notation_size)
    plt.annotate('SCI(GPU)', (64.32-60, 22.4 - 3.5), fontsize=notation_size)
    plt.annotate('ReconfigISP(GPU)', (89 - 70, 26.8 + 0.10), fontsize=notation_size)
    plt.annotate('AnscombeNet(GPU)', (123 - 70, 42.1 + 0.10), fontsize=notation_size)
    plt.annotate('Our(GPU)s', (32.7 - 70, 54.9 + 0.10), fontsize=notation_size)



    ############## for cpu time
    # HDRPlus ISP
    x = [2100]
    y = [49.5]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('HDR ISP', (2300 - 150, 49.5 + 1), fontsize=notation_size)

    # CLAHE, Manituke, SCI, ReconfigISP
    x = [660, 402, 1831, 1290]
    y = [50.5, 48.5, 22.4, 42.1]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=1.0, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('CLAHE', (270 - 70, 50.5 + 1), fontsize=notation_size)
    plt.annotate('Manituke', (302+30 , 48.5 + 1), fontsize=notation_size)


    plt.xlim(0, 3000, auto=True)
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
