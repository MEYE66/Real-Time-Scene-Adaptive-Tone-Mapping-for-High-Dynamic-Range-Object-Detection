import shutil
import os
import cv2
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    
    src_path = "/home/ligongzhe/data/RAWtiff"
    dst_path = "/home/ligongzhe/data/raws_val"
    image_path = glob(os.path.join(src_path,  "*.tiff"))
    night_json = []
    day_json = []

    for path in image_path:
        if path.startswith('/home/ligongzhe/data/RAWtiff/night'):
            night_json.append(path)
        # elif path.startswith('home/lgz/data/HDR_RAW/anno/day'):
        else:
            day_json.append(path)

    night_json = sorted(night_json, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))
    day_json = sorted(day_json, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))

    night_len = len(night_json)
    day_len = len(day_json)
    num = 1000
    train_files = night_json[:night_len - num] + day_json[:day_len - num]
    val_files = night_json[night_len - num:] + day_json[day_len - num:]


    
    for i in val_files:
        # src = os.path.join(src_path, i)
        # dst = os.path.join(dst_path, i)
        # dst = i.replace("")
        dst = i.replace("/home/ligongzhe/data/RAWtiff", "/home/ligongzhe/data/raws_val")
        shutil.copyfile(i, dst)
