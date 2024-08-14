import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from skimage.util import view_as_blocks
from scipy.signal import convolve, gaussian
from scipy.ndimage.filters import generic_filter
from scipy.stats import norm, beta


from pycocotools.coco import COCO



def rms_contrast(image):
    return np.std(image)


def weber_contrast(image):
    # luminance / avg luminance
    image = (image- image.min())/(image.max()-image.min()).astype(np.float32)
    avg_lum = np.exp(np.mean(np.log(1e-6+image)))
    # return np.mean(avg_image/avg_lum)
    return np.mean((image - avg_lum)/ avg_lum)


def michelson_contrast(image):
    # image = (image- image.min())/(image.max()-image.min()).astype(np.float32)
    min_val = np.min(image)+1e-6
    max_val = np.max(image)
    out = (max_val - min_val)/(max_val + min_val)
    return out



def tmqi_contrast(image, win=11):
    phat1 = 4.4
    phat2 = 10.1
    image = (image- image.min())/(image.max()-image.min()).astype(np.float32)
    image = image * 255.
    u = np.mean(image)
    
    W, H = image.shape
    
    w_extra = (11 - W % 11)
    h_extra = (11 - H % 11)
    
    # zero padding to simulate matlab's behaviour
    if w_extra > 0 or h_extra > 0:
        test = np.pad(image, pad_width=((0, w_extra), (0, h_extra)), mode='constant')
    else:
        test = image
    # block view with fixed block size, like in the original article
    view = view_as_blocks(test, block_shape=(11, 11))
    sig = np.mean(np.std(view, axis=(-1, -2)))
    
    
    beta_mode = (phat1 - 1.) / (phat1 + phat2 - 2.)
    C_0 = beta.pdf(beta_mode, phat1, phat2)
    C = beta.pdf(sig / 64.29, phat1, phat2)
    pc = C / C_0
    return pc





def count_dataset_contrast(file_list):
    for file in file_list:
        return 




def main():
    RAW_path = "/home/ligongzhe/data/RAWtiff"
    RGB_path = "/home/ligongzhe/data/RGB"
    CLAHE_path = "/home/ligongzhe/data/RGBhe"
    Zero_path = "/home/ligongzhe/data/zero_dcepp"
    
    Ours_path = "/home/ligongzhe/data/ours"
    RAOD_path = "/home/ligongzhe/data/raodnet"
    # IANet_path = "/home/ligongzhe/data/ianet"
    root_list = [RGB_path, CLAHE_path, Zero_path, RAOD_path, Ours_path]
    # root_list = [RAW_path]
    
    coco_api = COCO("/home/ligongzhe/data/annotations/rgb/val.json")
    image_paths = coco_api.getImgIds()


    for root_path in root_list:
        # dataset 
        contrast_metric = 0
        cnt = 0
        for i in image_paths:
            #images
            image_name = coco_api.loadImgs(i)[0]['file_name']
            # image_name = image_name.replace('tiff', 'png')
            image_path = os.path.join(root_path, image_name)
            try:
                # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                contrast_metric += weber_contrast(image)
                cnt +=1
            except:
                continue

            # contrast_metric += tmqi_contrast(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
  
        print(f"{root_path} contrast metric:{contrast_metric/cnt}, {cnt}")
        
    exit(234)
    
    
    
    data_id = "day-02000.png"
    data_id = "night-15000.png"
    hdr_isp_data = cv2.imread(os.path.join("/home/ligongzhe/data/RGB", data_id), cv2.IMREAD_GRAYSCALE)
    print(tmqi_contrast(hdr_isp_data))
    clahe_data = cv2.imread(os.path.join("/home/ligongzhe/data/RGBhe/", data_id), cv2.IMREAD_GRAYSCALE)
    print(tmqi_contrast(clahe_data))
    return 


if __name__ == '__main__':
    main()


