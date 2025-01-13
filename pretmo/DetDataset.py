import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from matplotlib import pyplot as plt




BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24

LoD_CLASSES_COLOR = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                     (128, 0, 128)]

def pltImg(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


class DetImageDataset(Dataset):
    def __init__(self,
                 train=True,
                 image_dir='/mnt/data1/RoD/train_test',
                 annot_dir='/mnt/data1/RoD/annotations',
                 transform=None,
                 Scene=None,
                 ):
        self.image_dir = image_dir
        self.annot_dir = annot_dir

        self.coco = COCO(self.annot_dir)
        self.image_ids = self.coco.getImgIds()

        if train:
            # filter image id without annotation
            ids = []
            for image_id in self.image_ids:
                annot_ids = self.coco.getAnnIds(imgIds=image_id)
                annots = self.coco.loadAnns(annot_ids)
                scene = self.coco.loadImgs(image_id)[0]['file_name'].split('-')[0]
                if Scene is not None and Scene != scene:
                    continue
                if len(annots) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids

        self.cat_ids = self.coco.getCatIds()
        self.cats = sorted(self.coco.loadCats(self.cat_ids),
                           key=lambda x: x['id'])
        self.num_classes = len(self.cats)

        # cat_id is an original cat id,coco_label is set from 0 to 79
        self.cat_id_to_cat_name = {cat['id']: cat['name'] for cat in self.cats}
        self.cat_id_to_coco_label = {
            cat['id']: i
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_id = {
            i: cat['id']
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_name = {
            coco_label: self.cat_id_to_cat_name[cat_id]
            for coco_label, cat_id in self.coco_label_to_cat_id.items()
        }

        self.transform = transform

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = self.load_tiff(index)
        # image = self.load_jpg(index)
        annotations = self.load_annots(index)
        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        
        
        hdr_name = self.coco.loadImgs(self.image_ids[index])[0]['file_name']
        sample = {
            'image': image,
            'annots': annotations,
            'scale': scale,
            'size': size,
            'image_val': None,
            'image_hsv': None,
            'hdr_name': hdr_name,

        }
        
        
            # image_val = sample_batched['image_val']
            # image_hsv = sample_batched['image_hsv']
            # hdr_name = sample_batched['hdr_name']
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
        
    def load_tiff(self, idx):
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        path = os.path.join(self.image_dir, file_name)
        # raw = np.load(path).astype(np.float32)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = (img - img.min())/(img.max() - img.min())
        img = img.astype(np.float32)
        # raw = np.clip(raw, 0, None)  
        return img


    def load_annots(self, idx):
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annots = self.coco.loadAnns(annot_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']
        targets = np.zeros((0, 5))
        if len(annots) == 0:
            return targets.astype(np.float32)
        # filter annots
        for annot in annots:
            if 'ignore' in annot.keys():
                continue
            # bbox format:[x_min, y_min, w, h]
            bbox = annot['bbox']  # bbox format:[xmin, ymin, width, height]
            inter_w = max(0, min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
            inter_h = max(0, min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
            if inter_w * inter_h == 0:
                continue
            if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
                continue
            if annot['category_id'] not in self.cat_ids:
                continue
            target = np.zeros((1, 5))
            target[0, :4] = bbox
            target[0, 4] = self.cat_id_to_coco_label[annot['category_id']]
            targets = np.append(targets, target, axis=0)

        # transform bbox targets from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        targets[:, 2] = targets[:, 0] + targets[:, 2] #
        targets[:, 3] = targets[:, 1] + targets[:, 3] #

        return targets.astype(np.float32)




if __name__ == '__main__':
    
    data_path = "/home/ligongzhe/data/RAWtiff/"
    
    json_path = "/home/ligongzhe/data/annotations/patch_res/val.json"
    
    

    

    dataset = DetImageDataset(root_dir=data_path, annot_dir=json_path, )
    print(dataset.__len__())
    data = dataset.__getitem__(7)
    
    
    
    image = data['image']

    print(f"image_hsv:{image.shape},  {image.min()}, {image.max()}")


    exit(234)