import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import pandas as pd
import pydicom

import os

class CTDataset(Dataset):

    def __init__(self,target_file_dir, img_dir,preprocessing_params,range1,range2,range3):
        self.range1 = range1
        self.range2 = range2
        self.range3 = range3
        self.labels = self.get_labels(target_file_dir)
        self.img_labels = pd.read_csv(target_file_dir)
        self.img_dir = img_dir
        self.preprocessing_params = preprocessing_params

    def __len__(self):
        return len(self.labels)
        #return (self.range[1]-self.range[0])*3

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.labels[index][0])
        #print(img_path)
        img = cv2.imread(img_path)
        img = self.process(img)
        img = torch.tensor(img)
        label = self.labels[index][1]
        sample = img, label
        return sample

    def process(self,img):
        t = self.preprocessing_params[0]
        i = self.preprocessing_params[1]
        k = self.preprocessing_params[2]
        x = self.preprocessing_params[3]
        y = self.preprocessing_params[4]
        w = self.preprocessing_params[5]
        h = self.preprocessing_params[6]
        r = self.preprocessing_params[7]
        
        cimg = img[y:y+h,x:x+w,:]
        rimg = cimg.copy()
        gimg = cv2.cvtColor(cimg.copy(),cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gimg.copy(), t,255,cv2.THRESH_BINARY)
        dilation = cv2.erode(th1.copy(),np.ones((k,k),np.uint8),iterations=i)
        contours, _ = cv2.findContours(dilation.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        l = 0
        
        for c in contours:
            if len(c)>l:
                l=len(c)

                maxY = max(c.T[1][0])
                minY = min(c.T[1][0])
                maxX = max(c.T[0][0])
                minX = min(c.T[0][0])
        resized = cv2.resize(rimg[minY:minY+(maxY-minY),minX:minX+(maxX-minX),:],(r,r))
        resized = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        return resized

    def get_labels(self,target_file_dir):
        labels = []
        df = pd.read_csv(target_file_dir)
        for i in df["filename"]:
            if i[:2] == "IN":
                if int(i[7:i.index(".")]) < self.range1[1] and int(i[7:i.index(".")]) >= self.range1[0]:
                    labels.append((i,0))

            elif i[:2] == "IS":
                if int(i[6:i.index(".")]) < self.range2[1] and int(i[6:i.index(".")]) >= self.range2[0]:
                    labels.append((i,1))

            elif i[:2] == "KA":
                if int(i[6:i.index(".")]) < self.range3[1] and int(i[6:i.index(".")]) >= self.range3[0]:
                    labels.append((i,2))
        return labels
            

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

class CTDataset2(Dataset):

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