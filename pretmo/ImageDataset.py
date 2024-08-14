
import os
import functools
import numpy as np
from torch.utils.data import Dataset
import random
import build_nlp
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib
import np_transforms

from pycocotools.coco import COCO

# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.exr', '.hdr']
# IMG_EXTENSIONS = ['.exr', '.hdr']
IMG_EXTENSIONS = ['.raw']

BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def pack_raw_image(image_raw):
    """ Packs a single channel bayer image into 4 channel tensor, where channels contain R, G, G, and B values"""
    # if isinstance(image_raw, np.ndarray):
    #     im_out = np.zeros((image_raw.shape[0] // 2, image_raw.shape[1] // 2, 4), dtype=image_raw.dtype)
    # elif isinstance(image_raw, torch.Tensor):
    #     im_out = torch.zeros((image_raw.shape[0] // 2, image_raw.shape[1] // 2, 4), dtype=image_raw.dtype)
    # else:
    #     raise Exception("Invalid input type. Expected numpy ndarray or torch Tensor.")

    im_out = np.zeros((image_raw.shape[0] // 2, image_raw.shape[1] // 2, 3), dtype=image_raw.dtype)

    im_out[:, :, 0] = image_raw[0::2, 0::2] # r
    im_out[:, :, 1] = image_raw[0::2, 1::2] # g
    # im_out[:, :, 2] = image_raw[1::2, 0::2] # g
    im_out[:, :, 2] = image_raw[1::2, 1::2] # b
    return im_out


def apply_awb(image):
    mean_r = np.mean(image[:, :, 0]) # r
    mean_g = np.mean(image[:, :, 1]) # g
    mean_b = np.mean(image[:, :, 2]) # b
    image[:, :, 0] *= mean_g / mean_r
    image[:, :, 2] *= mean_g / mean_b
    return image

# def image_loader(image_name):
#     if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
#         #img = imageio.imread(image_name)
#         img = cv2.imread(image_name)
#     return img


def minmax_norm(img):
    img = (img - img.min())/(img.max()-img.min())
    img = img.astype(np.float32)
    return img



def raw_loader(image_name):
    raw = np.fromfile(image_name, dtype=np.uint8)
    raw = raw.reshape(1856, 2880, 3).astype(np.float32)
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)  # shape [1856, 2880, 1]
    # return raw / (BIT24 - 1)  # norm to range [0, 1]
    raw = (raw - np.min(raw))/(np.max(raw)-np.min(raw)).astype(np.float32)
    raw = np.clip(raw, 0, 1.)
    raw = np.squeeze(raw)
    raw = pack_raw_image(raw)
    raw = apply_awb(raw)
    return raw



def tiff_load(image_name):
    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    img = minmax_norm(img)
    return img


def get_default_img_loader():
    # return functools.partial(png_load)
    # return functools.partial(npy_load)
    return functools.partial(tiff_load)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)



class ImageDataset(Dataset):
    def __init__(self,
                 img_dir,
                 annot_dir=None,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader):
        self.img_dir = img_dir
        self.test = test
        self.loader = get_loader()

        self.transform = transform
        
        if annot_dir:
            self.coco = COCO(annot_dir)
            self.paths = self.coco.getImgIds()
        else:
            self.paths = list(scandir(img_dir, recursive=True))

        # with open(img_dir, "r") as file:
            # lines = file.readlines()
        # self.paths = [i.replace("\n","") for i in lines]
        
    def __getitem__(self, index):
        
        file_name = self.coco.loadImgs(self.paths[index])[0]['file_name']
        image_name = os.path.join(self.img_dir, file_name)
        # image_name = os.path.join(self.img_dir, self.paths[index])
        image_rgb = self.loader(image_name) 
        
        # image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        # if self.transform is not None:            
        #     image_hsv = self.transform(image_hsv)  # self.transform(hdr_h[:, :, 2][:, :, np.newaxis])
        #     image_val = image_hsv[2:3, :, :, ]
        
        if self.transform is not None:
            image_rgb = self.transform(image_rgb)
        
        
        # if self.test:
        #     s_min = 5.0
        #     s_max = 1e6
        #     # hdr_h = torch.from_numpy(hdr_h)
        # else:
        #     s_min = 5.0
        #     s_max = random.choice([1e4,1e5,1e6,1e7]) #1e7 1e8  1e9
        #     # hdr_h = self.transform(hdr_h)
        # # s_max = 1e6
        # # s_min = 5.0
        # # s_max = torch.max(image_val)/torch.exp(torch.mean(torch.log(1e-6+image_val)))
        # image_val = ((image_val - image_val.min()) / (image_val.max() - image_val.min()))
        # image_val = (s_max - s_min) * image_val + s_min
        sample = {'image_val': image_rgb, 'image_hsv': image_rgb, 'hdr_name': image_name}
        return sample


    def __len__(self):
        return len(self.paths)

    def to_tensor(self,np_image):
        return torch.FloatTensor(np_image.transpose((2, 0, 1)).copy())

    def Resize(self,np_image):
        n_t = np_transforms.Scale(512)
        return n_t(np_image)



def tensor2np(tensor):
    return tensor.permute(1,2,0).cpu().numpy()


def pltImg(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()



def main():
    # data_path = "/home/lgz/workspace/TMO_CAN-master/data/raw_input/HDR_RAW/day-02000.raw"
    # raw = np.fromfile(data_path, dtype=np.uint8)
    # raw = raw.reshape(1856, 2880, 3).astype(np.float32)
    # raw = np.split(raw, 3, axis=2)
    # raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)  # shape [1856, 2880, 1]
    # # return raw / (BIT24 - 1)  # norm to range [0, 1]
    # raw = (raw - np.min(raw)) / (np.max(raw) - np.min(raw)).astype(np.float32)
    # raw = np.clip(raw, 0, 1.)
    # raw = np.squeeze(raw)
    # raw = pack_raw_image(raw)
    # raw = raw.astype(np.float32)
    # value = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV)[:,:,2]
    # # pltImg(value)

    data_path = "/home/ligongzhe/data/RAW/"
    value = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
    # pltImg(value)
    value = minmax_norm(value)
    value = torch.from_numpy(value).cuda()
    value = value.unsqueeze(0).unsqueeze(0)
    print(value.shape)

    nlp = build_nlp.nlpclass()
    nlp = nlp.nlp(value)
    for i in range(nlp.__len__()):
        nlp[i] = nlp[i].squeeze().unsqueeze(0)
        # print(nlp[i].shape)
        out = nlp[i].squeeze(0).cpu().numpy()
        pltImg(out)
        
        
if __name__ == '__main__':
    data_path = "/home/ligongzhe/data/RAWtiff/"
    json_path = "/home/ligongzhe/data/annotations/tmp/val.json"
    tf = np_transforms.Compose([
        # np_transforms.RandomCrop(512),
        # np_transforms.RandomHorizontalFlip(),
        np_transforms.ToTensor()
    ])
    dataset = ImageDataset(data_path, json_path, transform=tf, test=False)
    
    print(dataset.__len__())
    data = dataset.__getitem__(7)
    image_val = data['image_val']
    image_hsv = data['image_hsv']
    hdr_name = data['hdr_name']
    print(f"image_val:{image_val.shape},  {image_val.min()}, {image_val.max()}")
    # print(f"image_hsv:{image_hsv.shape},  {image_hsv.min()}, {image_hsv.max()}")

    exit(234)

    # print(gI.shape)
    # # print(torch.isnan(i) for i in nlp)
    # # print(torch.isnan(gI))
    # out = tensor2np(gI)
    # pltImg(out)
    # ['/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105133323484_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105133914692_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105140053363_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105133025764_image0.png']

    # path_list = ['/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105133818622_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105151651269_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105144109674_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105144124709_image0.png']

    # path_list = [
    #     '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105150648075_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105144144753_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105134348800_image0.png',
    #  '/home/lgz/data/RhoVision/official_isp/LUCID_TRI054S-C_223800455__20231105151747093_image0.png'
    #              ]
    # path_list = [
    #     '/home/lgz/data/RhoVision/tmp/LUCID_TRI054S-C_223800455__20231105150648075_image0.npy',
    #  '/home/lgz/data/RhoVision/tmp/LUCID_TRI054S-C_223800455__20231105144144753_image0.npy',
    #  '/home/lgz/data/RhoVision/tmp/LUCID_TRI054S-C_223800455__20231105134348800_image0.npy',
    #  '/home/lgz/data/RhoVision/tmp/LUCID_TRI054S-C_223800455__20231105151747093_image0.npy'
    #              ]

    # labelme_json = glob.glob('/home/lgz/data/RoD/RAWo/*.raw')
    # night_json = []
    # day_json = []
    # for path in labelme_json:
    #     if path.startswith('/home/lgz/data/RoD/RAWo/night'):
    #         # if path.startswith('/Tompus_Sun/Dataset/Annotations/night'):
    #         night_json.append(path)
    #     # elif path.startswith('/Tompus_Sun/Dataset/Annotations/day'):
    #     elif path.startswith('/home/lgz/data/RoD/RAWo/day'):
    #         day_json.append(path)
    #
    # night_json = sorted(night_json, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))
    # day_json = sorted(day_json, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))
    #
    # # sorted_files = sorted(labelme_json, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) # Tompus_Sun/Dataset/Annotations
    #
    # night_len = len(night_json)
    # day_len = len(day_json)
    # num = 1000
    # train_files = night_json[:night_len - num] + day_json[:day_len - num]
    # val_files = night_json[night_len - num:] + day_json[day_len - num:]
    # # print(len(val_files))
    #
    #
    # train_txt = '/home/lgz/data/RoD/train.txt'
    # val_text = '/home/lgz/data/RoD/val.txt'
    #
    # # Open the file in write mode
    # with open(val_text, "w") as file:
    #     for idx in (val_files):
    #         file.write(f'{idx}\n')
    # # Open the file in read mode
    # with open(val_text, "r") as file:
    #     # Read all lines from the file
    #     lines = file.readlines()
    # print(lines)

















