from pandas.core.base import SelectionMixin
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd

import cv2
import os
import pydicom
import numpy as np

class CTDataset(Dataset):

    def __init__(self,target_file_dir, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(target_file_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        img = torch.tensor(cv2.imread(img_path,0))
        label = self.img_labels.iloc[index,1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        sample = img,label
        return sample

class DCMDataset(Dataset):
    
    def __init__(self,target_file_dir, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(target_file_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        img = pydicom.read_file(img_path)
        img.pixel_array.dtype=np.int16
        img = torch.tensor(img.pixel_array)
        label = self.img_labels.iloc[index,1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        sample = img,label
        return sample



"""
class DCMDataset(Dataset):

    def __init__(self,X_dir,y_dir,X_transform=None, y_transform=None):

        self.X_dir = X_dir
        self.y_dir = y_dir
        self.X_transform = X_transform
        self.y_transform = y_transform
        self.X = torch.load(X_dir)
        self.y = torch.load(y_dir)
        
    def __len__(self):
        return len(self.y)

    def get_item(self, index):
        return self.X[index],self.y[index]

"""