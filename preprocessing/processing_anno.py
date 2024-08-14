import os
import json
import shutil
import cv2


# 替换关键字
def replace_keywords(text, replacements):
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def replace_json_name():
    with open('/home/ligongzhe/data//annotations/scene/raw/night/val.json', 'r') as f:
        json_str = f.read()

    data = json.loads(json_str)
    # print(data['images'])
    for image in data['images']:
        filename = image['file_name']
        if filename.endswith('.tiff'):
            image['file_name'] = filename.replace('tiff', 'png')
            
    # 写入到新的.json文件
    with open('/home/ligongzhe/data//annotations/scene/rgb/night/val.json', 'w') as f:
        json.dump(data, f)


def replace_file_name():
    root = '/home/lgz/data/HDR_RAW/ldr_val'

    for filename in os.listdir(root):
        if filename.endswith(".npy.png"):
            # 构造新文件名
            new_filename = filename.replace(".npy.png", ".png")
            # 重命名文件
            os.rename(os.path.join(root, filename), os.path.join(root, new_filename))
    print("Done")
    return



# def tmp():
#     src_path = '/home/lgz/data/HDR_RAW/rgbs_val/'
#     # root = '/home/lgz/data/HDR_RAW/scene/night'
#     day_dst_path = '/home/lgz/data/HDR_RAW/scene/rgbs_val/day'
#     night_dst_path = '/home/lgz/data/HDR_RAW/scene/rgbs_val/night'
#
#     for filename in os.listdir(src_path):
#         base_name = filename.split('-')[0]
#         if base_name == 'night':
#             os.symlink(os.path.join(src_path, filename), os.path.join(night_dst_path, filename))
#         elif base_name =='day':
#             os.symlink(os.path.join(src_path, filename), os.path.join(day_dst_path, filename))


if __name__ == '__main__':

    # data/annotations/scene/day
    # data/annotations/scene/night
    # generate raw json
    replace_json_name()



#     #### generate scene json
#     json_path = '/home/lgz/data/HDR_RAW/anno/'
#     src_path = '/home/lgz/data/HDR_RAW/scene/raws_val/day'
#     dst_path = '/home/lgz/data/HDR_RAW/scene/raws_val/day_anno'
# #     day_dst_path = '/home/lgz/data/HDR_RAW/scene/rgbs_val/day'
# #     night_dst_path = '/home/lgz/data/HDR_RAW/scene/rgbs_val/night'
# #
#     for filename in os.listdir(src_path):
#         base_name = filename.split('.')[0]
#         # print(base_name)
#         # json_file = os.path.join(json_path, base_name)
#         shutil.copy(os.path.join(json_path, base_name+'.json'), os.path.join(dst_path,  base_name+'.json'))



