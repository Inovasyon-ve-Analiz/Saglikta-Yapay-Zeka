import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import os

class CTDataset(Dataset):

    def __init__(self, data_dir, binary_classification, mode):
        self.mode = mode
        self.data = []
        self.transform = A.Compose([ 
             A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
        ])
        
        for i,f in enumerate(os.listdir(data_dir)):
            if f[:2] == "IS":
                self.data.append((os.path.join(data_dir, f), 0))
            elif f[:2] == "KA":
                self.data.append((os.path.join(data_dir, f), 1))
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img_path = self.data[index][0]
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (512,512))
        if self.mode == "train":
            img = self.transform(image=img)["image"]

        img = torch.tensor(img)
        label = self.data[index][1]
        sample = img, label
        return sample
