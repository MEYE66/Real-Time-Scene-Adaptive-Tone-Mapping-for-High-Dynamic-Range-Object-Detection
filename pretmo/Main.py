import argparse
import TrainModel
import os
import torch

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--trainset", type=str, default="/home/ligongzhe/data/raws_val/")
    parser.add_argument("--testset", type=str, default="/home/ligongzhe/data/raws_val/")
    parser.add_argument("--test_anno", type=str, default="/home/ligongzhe/data/annotations/patch_res/val.json")
    # parser.add_argument("--trainset", type=str, default="/home/lgz/data/RhoVision/official_isp/")
    # parser.add_argument("--testset", type=str, default="/home/lgz/data/RhoVision/official_isp/") #rgb_tmp
    # parser.add_argument("--trainset", type=str, default='/home/lgz/data/RoD/val.txt') # rgb_tmp
    # parser.add_argument("--testset", type=str, default='/home/lgz/data/ RoD/val.txt') # rgb_tmp

    parser.add_argument("--results_savepath", type=str,
                        default="/home/ligongzhe/data/ours/")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument('--ckpt_path', default='/home/ligongzhe/mmdetection/experiments/raw/faster-rcnn_logscan_1e4_1e7/', type=str,
                          metavar='PATH', help='path to checkpoints')  # ours tmo ckpt
    # parser.add_argument('--ckpt_path', default='/home/ligongzhe/mmdetection/arch_experiment/compare/faster-rcnn_raodnet/', type=str,
    #                       metavar='PATH', help='path to checkpoints') # raodnet tmo ckpt
    # parser.add_argument('--ckpt_path', default='/home/ligongzhe/mmdetection/arch_experiment/compare/faster-rcnn_ianet/', type=str,
    #                       metavar='PATH', help='path to checkpoints') # ianet tmo ckpt
    # /home/ligongzhe/mmdetection/arch_experiment/compare/faster-rcnn_ianet/epoch_13.pth
    # parser.add_argument('--ckpt', default='ada_canlog-00019.pt', type=str, help='name of the checkpoint to load')
    # epoch_48.pth'
    
    # /home/ligongzhe/mmdetection/arch_experiment/compare/faster-rcnn_raodnet/epoch_48.pth 
    parser.add_argument('--ckpt', default='epoch_13.pth', type=str, help='name of the checkpoint to load')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512, help='None means random resolution')

    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay_interval", type=int, default=9)
    parser.add_argument("--decay_ratio", type=float, default=0.1)

    parser.add_argument("--epochs_per_eval", type=int, default=10)
    parser.add_argument("--epochs_per_save", type=int, default=10)

    return parser.parse_args()


def main(cfg):
    t = TrainModel.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        t.eval()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    config = parse_config()

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
        
    if not os.path.exists(config.results_savepath):
        os.makedirs(config.results_savepath)
    main(config)





