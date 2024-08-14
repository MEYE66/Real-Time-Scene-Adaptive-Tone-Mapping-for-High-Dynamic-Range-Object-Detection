import os
import glob
import json
import shutil

import cv2
from parse_anno import Lableme2CoCo

def main():
    json_path = "/home/ligongzhe/data/archive/00Train-anno/"
    # labelme_json=glob.glob('/mnt/data1/RoD/00Anno/*.json')
    json_path = glob.glob("/home/ligongzhe/data/archive/00Train-anno/*.json")
    night_json = []
    day_json = []
    
    


    for path in json_path:
        if path.startswith('/home/ligongzhe/data/archive/00Train-anno/night'):
            night_json.append(path)
        elif path.startswith('/home/ligongzhe/data/archive/00Train-anno/day'):
            day_json.append(path)

    night_json = sorted(night_json, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))
    day_json = sorted(day_json, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))

    night_len = len(night_json)
    day_len = len(day_json)
    num = 1000
    
    # night
    # scene_train_files = night_json[:night_len - num]
    # scene_val_files =  night_json[night_len - num:]
    # day
    scene_train_files = day_json[:day_len - num]
    scene_val_files = day_json[day_len - num:]
        
    dst_path = "/home/ligongzhe/data/annotations/scene/day/"

    label2coco = Lableme2CoCo(img_postfix='.tiff')
    instance = label2coco.to_coco(scene_train_files)
    label2coco.save_coco_json(instance, dst_path + 'train.json')
    
    label2coco = Lableme2CoCo(img_postfix='.tiff')
    instance = label2coco.to_coco(scene_val_files)
    label2coco.save_coco_json(instance, dst_path + 'val.json')

if __name__ == '__main__':
    # data/annotations/scene/day
    # data/annotations/scene/night
    # generate raw json
    main()