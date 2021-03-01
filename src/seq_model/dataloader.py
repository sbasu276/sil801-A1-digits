from __future__ import print_function, division
import cv2
import os
import torch
import json
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "X"]

#df = {"<filename>": ["label1", "label2", ...]}

class DigitsDataset(Dataset):
    def __init__(self, img_dir, labels, img_set, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        d = []
        for key in img_set:
            for i in range(16):
                val = {}
                fname = key[:-4]+"_%s.png"%i
                if fname in labels.keys():
                    seq_labels = []
                    for e in labels[fname]:
                        seq_labels.append(LABELS.index(e))
                    val["label"] = seq_labels
                    val["filename"] = fname
                    d.append(val)
        self.df = d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        # Transforms
        if self.transforms:
            img = self.transforms(image)
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        return img, label, filename


if __name__ == "__main__":
    img_transforms = T.Compose([T.ToTensor()])
    d = {}
    with open("data/row_labels.txt","r") as f:
        for e in f.readlines():
            k, v = e.strip().split(",")
            d[k] = v
    dataset = DigitsDataset("data/row_imgs", d, ["1501.png","1503.png"], transforms = img_transforms)
    loader = DataLoader(dataset, batch_size=5, collate_fn=utils.collate_fn)
    for im, lb, f in loader:
        print(f, lb, im[0].shape)
        #images, labels, filename = next(iter(loader))
    #print(labels)
