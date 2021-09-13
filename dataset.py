import torch
from torch.utils.data import Dataset
import cv2
import os

class CTDataset(Dataset):

    def __init__(self, data_dir, binary_classification):

        self.data = []

        for i,f in enumerate(os.listdir(data_dir)):
            if f[:2] == "IN":
                self.data.append((os.path.join(data_dir, f), 0))
            elif f[:2] == "IS":
                self.data.append((os.path.join(data_dir, f), 1))
            elif f[:2] == "KA":
                if binary_classification:
                    self.data.append((os.path.join(data_dir, f), 1))
                else:
                    self.data.append((os.path.join(data_dir, f), 2))
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img_path = self.data[index][0]
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (512,512))
        img = torch.tensor(img)
        label = self.data[index][1]
        sample = img, label
        return sample
