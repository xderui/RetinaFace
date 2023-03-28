import torch
import torch.utils.data as data
import numpy as np
import cv2
from data.data_augment import preproc
from data.config import cfg_mnet
cfg = cfg_mnet

class ufddface(data.Dataset):
    def __init__(self,txt_path):
        self.imgpath = []
        self.target = []
        f = open(txt_path,'r')
        lines = f.readlines()
        lines_len = len(lines)
        idx = 0
        while idx < lines_len:
            img_path = str(txt_path.replace('UFDD_val_bbx_gt_woDistractor.txt','images/') + lines[idx])
            img_path = img_path.strip('\n')
            self.imgpath.append(img_path)
            idx += 1
            tmp_target = []
            for j in range(int(lines[idx])):
                line = lines[idx+j+1].split(' ')
                line_float = [float(x) for x in line[:4]]
                line_float[2] = line_float[0] + line_float[2]
                line_float[3] = line_float[1] + line_float[3]
                line_float.append(1)
                tmp_target.append(line_float)
            idx += int(lines[idx])+1
            self.target.append(tmp_target)

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, index):
        img = cv2.imread(self.imgpath[index])

        # print(img.shape)
        annotations = np.zeros((0, 5))
        # print(self.target[index])
        for idx, bbox in enumerate(self.target[index]):
            annotation = np.zeros((1, 5))
            annotation[0, 0] = bbox[0]
            annotation[0, 1] = bbox[1]
            annotation[0, 2] = bbox[2]
            annotation[0, 3] = bbox[3]
            annotation[0, 4] = bbox[4]

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)
        # print(target)
        processor = preproc(cfg['image_size'], (104, 117, 123))
        img, target = processor(img,target)
        # print(img.shape)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

#
# import torch.utils.data as data
#
# img = cv2.imread('UFDD/images/illumination/illumination_00337.jpg')
# print(img.shape)
#
# dataset = ufddface('UFDD/UFDD_val_bbx_gt_woDistractor.txt')
# batch_iterator = iter(data.DataLoader(dataset, 2, shuffle=True, num_workers=0, collate_fn=detection_collate))
# images, targets = next(batch_iterator)
# print(images.shape,targets.shape)
#
#
#
