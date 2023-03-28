import cv2
import torch.utils.data as data
from ufdd_face import ufddface
import torch
import numpy as np


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


dataset = ufddface('./data/UFDD/UFDD_val_bbx_gt_woDistractor.txt')
img = cv2.imread('./data/UFDD/images/illumination/illumination_00383.jpg')
# print(img.shape)
dataloader = iter(data.DataLoader(dataset,2, shuffle=True, num_workers=0, collate_fn=detection_collate))
img,target = next(dataloader)

print(img.shape,len(target))
