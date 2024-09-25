# coding: utf-8
# File: patch_vis.py
# Description: Numpy helpers for image processing
# Created: 2024-09-09  
# Author: Gongzhe Li
import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"




def pltImg(img):
    plt.figure()
    # plt.imshow(img, cmap='gray')
    plt.imshow(img)
    plt.show()


def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1
    return img


def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)
    return img

def drawrect(img,pt1,pt2,color,thickness=5,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    img = drawpoly(img,pts,color,thickness,style)
    return img


def get_image_crop(image, crop_region):

    image_crop = image[crop_region[0][1]:crop_region[1][1], crop_region[0][0]:crop_region[1][0]]
    # image_crop = image
    # image_with_rect = cv2.rectangle(image, crop_region[0], crop_region[1], [255,0,0], 2)
    image_with_rect = drawrect(image, crop_region[0], crop_region[1], [0,0,255], thickness=3, style='1')
    # image_with_rect = drawrect(image, crop_region[0], crop_region[1], [102,0,51], thickness=2, style='1')

    # pltImg(image)
    # pltImg(image_with_rect)
    return image_crop, image_with_rect


if __name__ == '__main__':
    # night-15581.png    [(700, 300), (900, 500)]
    # night-15648.png
    # night-15594.png
    root_path = "/home/gongzheli/workspace/HDRPreprocessor-Detection/visualization/det_results/"
    # root_path = "/home/gongzheli/workspace/supp/day/2/"
    method_paths = ['ours', 'isp', 'clahe', 'raodnet', 'zero', 'man']
    # image_name = "night-15581.png"
    # crop_region = [(700, 300), (900, 500)]

    # image_name = "day-05137.png"
    # crop_region = [(270, 300), (470, 500)]

    # image_name = "night-15648.png"
    # crop_region = [(1080, 500), (1280, 700)]

    # image_name = "day-05267.png"
    # crop_region = [(250, 500), (450, 700)]   # 200x200

    image_name = "day-05825.png"
    crop_region = [(300, 350), (500, 550)]   # 200x200

    # image_name = "night-15624.png"
    # crop_region = [(700, 350), (900, 550)]


    # image_name = "night-15745.png"
    # crop_region = [(300, 550), (500, 750)]


    for i in method_paths:
        image_path = os.path.join(root_path, i, image_name)
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # pltImg(image)
        image_crop, image_with_rect = get_image_crop(image, crop_region)
        # print(image.shape)
        # pltImg(image_crop)
        # pltImg(image_with_rect)
        # exit(234)

        cv2.imwrite(f"{os.path.join(root_path, i, 'rect-'+image_name)}.png", image_with_rect)
        cv2.imwrite(f"{os.path.join(root_path, i, 'crop-'+image_name)}.png", image_crop)


    # path = "./det_results/isp/night-15581.png"
    # image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # image_crop, image_with_rect = get_image_crop(image, crop_region)


    # print(os.path.basename(path))
    # pltImg(image_crop)
    # pltImg(image_with_rect)







