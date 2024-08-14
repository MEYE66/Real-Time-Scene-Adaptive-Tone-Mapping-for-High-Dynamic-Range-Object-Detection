import os
import time
from functools import reduce
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
import np_transforms
from E2etmo import E2ETMO, ResTMO, logCAN
from det_tmo import logSCANTMO, IANet, RAODNet
from ImageDataset import ImageDataset
from DetDataset import DetImageDataset

from NLPD import NLPD_Loss
from Gdn import Gdn2d, Gdn1d
import time


from torchvision import utils as vutils
import cv2

from matplotlib import pyplot as plt
from tqdm import tqdm


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.train_batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.results_savepath = config.results_savepath

        self.train_transform = np_transforms.Compose([
            # np_transforms.DownScale(config.image_size),
            # np_transforms.RandomCrop(config.image_size),
            # np_transforms.RandomHorizontalFlip(),
            np_transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
            # np_transforms.RandomCrop(config.image_size),
            np_transforms.ToTensor()
        ])
        self.train_data = ImageDataset(
                                    img_dir=config.trainset,
                                    annot_dir=config.test_anno,
                                    transform=self.train_transform,
                                    test=False)
        # self.train_data = DetImageDataset(
        #                             train=True,
        #                             image_dir=config.trainset,
        #                             annot_dir=config.test_anno,
        #                                )
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=8)
        # testing set configuration
        self.test_data = ImageDataset(
                                      img_dir=config.testset,
                                        annot_dir=config.test_anno,
                                      transform=self.test_transform,
                                      test=True)
        # self.test_data = DetImageDataset(
        #                             train=False,
        #                             image_dir=config.testset,
        #                             annot_dir=config.test_anno,
        #                             )
        self.test_loader = DataLoader(
                                    self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=16,
                                      )        


        self.device = torch.device("cuda:5" if torch.cuda.is_available() and config.use_cuda else "cpu")

        # self.model = E2ETMO(config.layer)
        # self.model = ResTMO(config.layer)
        # self.model = logCAN(config.layer)
        
        ### detection model
        self.model = logSCANTMO() # our model
        # self.model = IANet()
        self.model = RAODNet()
        
        
        self.model.to(self.device)

        # writer = SummaryWriter()
        self.model_name = type(self.model).__name__
        # loss function
        self.loss_fn = NLPD_Loss()

        self.loss_fn.to(self.device)
        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=lr,
                                    weight_decay=1e-4)


        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results = []
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        # if config.resume or not config.train:
        #     if config.ckpt:
        #         ckpt = os.path.join(config.ckpt_path, config.ckpt)
        #     else:
        #         ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
        #     if not ckpt:
        #         pass
        #     else:
        #         self._load_checkpoint(ckpt=ckpt)
        # if config.resume or not config.train:
        #     self.model = load_weight(self.model, os.path.join(config.ckpt_path, config.ckpt))
        self.model = load_weight(self.model, os.path.join(config.ckpt_path, config.ckpt))


        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)


    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch_loss = []
            _ = self._train_single_epoch(epoch)
            #print(123)
        # writer.close()

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('[*] Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
            if step < self.start_step:
                continue
            self.model.train()
            image_val = sample_batched['image_val']
            image_hsv = sample_batched['image_hsv']

            # for index in range(x.__len__()):
                # x[index] = x[index].to(self.device)
            image_val = image_val.to(self.device)
            y = self.model(image_val)
            
            # print(y.min(), y.max())
            # img_save = hdr[0,:,:,].squeeze().detach().cpu().numpy()
            # plt.figure()
            # plt.imshow(img_save)
            # plt.show()
            # print(y.min(), y.max())

            self.optimizer.zero_grad()
            self.loss = self.loss_fn(image_val, y)
            self.loss.backward()

            # print(sample_batched['hdr_name'])
            # for i in hdr:
            #     img = i.squeeze().detach().cpu().numpy()
            #     plt.figure()
            #     plt.imshow(img)
            #     plt.show()

            # grads = {}
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         grads[name] = torch.mean(param.grad)
            # print(grads)
            self.optimizer.step()
            self._gdn_param_proc()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, self.loss,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()
            self.epoch_loss.append(float(self.loss.data.cpu().data))
            # exit(234)


        self.train_loss.append(loss_corrected)
        self.scheduler.step()

        if (epoch+1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            self.model.eval()
            test_results = self.eval()
            self.test_results.append(test_results)
            out_str = 'Epoch {} Testing: NLPD: {:.4f}'.format(epoch, test_results)
            #writer.add_scalar('Test/Loss', test_results, epoch)
            print(out_str)

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results': self.test_results,
            }, model_name)

        return self.loss.data.item()

    def eval(self):
        nlpd_score = []
        for step, sample_batched in tqdm(enumerate(self.test_loader, 0)):

            image_val = sample_batched['image_val']
            image_hsv = sample_batched['image_hsv']
            hdr_name = sample_batched['hdr_name']

            start_time = time.time()
            # for index in range(x.__len__()):
            #     x[index] = x[index].to(self.device)
            image_val = image_val.to(self.device)
            y = self.model(image_val)
            
            stop_time = time.time()

            self._save_image(y, self.results_savepath, hdr_name[0], image_hsv)
            loss = self.loss_fn(image_val[:, 0:1, :, :,], y[:, 0:1, :, :, ])
            # loss = torch.Tensor(0)

            # print(hdr_name[0]+' '+str(float(loss)) + ' ' + str(stop_time-start_time))
            nlpd_score.append(float(loss.data.cpu().data))
        return reduce(lambda l1, l2: l1 + l2, nlpd_score) / len(nlpd_score)

    def _gdn_param_proc(self):
        for m in self.model.modules():
            if isinstance(m, Gdn2d) or isinstance(m, Gdn1d):
                m.beta.data.clamp_(min=2e-10)
                m.gamma.data.clamp_(min=2e-10)
                m.gamma.data = (m.gamma.data + m.gamma.data.t()) / 2


    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt)) 
            checkpoint = torch.load(ckpt)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.test_results = checkpoint['test_results']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        if len(all_times) == 0 :
            print('[*] No found checkpoint')
            return False
        else:
            return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)



    def save_image_tensor(self,input_tensor: torch.Tensor, filename):
     
        assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    
        input_tensor = input_tensor.clone().detach()
     
        input_tensor = input_tensor.to(torch.device('cpu'))
     
        vutils.save_image(input_tensor, filename)
        
        
        
    
    def _save_image(self, image_rgb, path, name, image_hsv=None):
    
        image_rgb = image_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()  # [H,W]
        image_rgb = (image_rgb-image_rgb.min())/(image_rgb.max()-image_rgb.min())
        image_rgb = np.clip(image_rgb*255, 0, 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_name = os.path.basename(name)
        img_path = os.path.join(path, image_name).replace("tiff", "png")
        # print(f"save path :{img_path}, save name:{name}")
        flag = cv2.imwrite(img_path, image_rgb)
        # print(f"save flag:{flag}, image shape:{image_rgb.shape},  save path:{img_path}")
    
    

    # def _save_image(self, image_val, path, name, image_hsv):
    #     # color
    #     self.d_max = 300
    #     self.d_min = 5
        
    #     image_val = image_val.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()  # [H,W]
    #     image_hsv = image_hsv.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()  # [H,W]
        
    #     image_val[image_val > self.d_max] = self.d_max
    #     image_val[image_val < self.d_min] = self.d_min
    #     image_val = (image_val - self.d_min) / (self.d_max - self.d_min)
        
    #     # gamma correction
    #     image_val = (image_val ** (1/2.2))
    #     image_val = (image_val-image_val.min())/(image_val.max()-image_val.min())
        
    #     # only save image_val (luminance channel)
    #     # image_val = np.clip(image_val * 255, 0, 255).astype(np.uint8)
    #     # cv2.imwrite(path + name.split('/')[-1] + '.png', image_val)

    #     # save image rgb
    #     image_hsv[:, :, 2:3] = image_val
    #     image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    #     image_rgb = (image_rgb-image_rgb.min())/(image_rgb.max()-image_rgb.min())
    #     image_rgb = np.clip(image_rgb*255, 0, 255).astype(np.uint8)
    #     # cv2.imwrite(path + name.split('/')[-1] + '.png', image_rgb)
    #     # print(name.replace('.tiff', '.png'))
    #     name = os.path.basename(name).replace('tiff', 'png')
    #     img_path = os.path.join(path, name)
    #     cv2.imwrite(img_path, image_rgb)


def load_weight(model, weight_path):
    weight = torch.load(weight_path)['state_dict']
    new_state_dict = {}
    for key, value in weight.items():
        if key.startswith('backbone.isp_preprocessor.'):
            new_key = key.replace('backbone.isp_preprocessor.', '', 1)  # 只替换第一个出现的旧前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value  # 其他情况不做更改
    
    # print(new_state_dict)
    # exit(234)
    model.load_state_dict(new_state_dict, strict=False)
    return model