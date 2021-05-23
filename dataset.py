import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import random
from torch.utils.data import random_split

class CTDataset(Dataset):
    random.seed(1)
    def __init__(self, folders, preprocessing_params, is_cropping, is_train, ratio):
        self.folders = folders
        self.preprocessing_params = preprocessing_params
        self.is_cropping = is_cropping
        self.labels = []
        self.ratio = ratio
        self.is_train = is_train

        image_dirs = []
        labels = []
        for dir in self.folders:
             for i,f in enumerate(os.listdir(dir)):
                image_dirs.append(os.path.join(dir,f))
                if f[:2] == "IN":
                   labels.append(0)
                elif f[:2] == "IS":
                   labels.append(1)
                elif f[:2] == "KA":
                   labels.append(2)

        mylist = np.array(list((zip(image_dirs, labels))))
        labels = np.array(labels, dtype=int)
        train_list = []
        test_list = []

        for i in range(3):
            list2 = mylist[labels == i]
            split = int(len(list2) * ratio)
            train_image_dirs, test_image_dirs = random_split(list2, [split, len(list2) - split],
                                    generator=torch.Generator().manual_seed(42))
            train_list.extend(train_image_dirs)
            test_list.extend(test_image_dirs)

        self.data = train_list if is_train == 1 else test_list
        #self.number_of_samples = len(os.listdir(image_dirs[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index][0]
        #print(img_path)
        img = cv2.imread(img_path,0)
        if self.is_cropping:
            img = self.process(img)
        else:
            img = cv2.resize(img,(512,512))
        img = torch.tensor(img)
        label = self.data[index][1]
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
        
        cimg = img[y:y+h,x:x+w]
        rimg = cimg.copy()
        #gimg = cv2.cvtColor(cimg.copy(),cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(rimg.copy(), t,255,cv2.THRESH_BINARY)
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
        resized = cv2.resize(rimg[minY:minY+(maxY-minY),minX:minX+(maxX-minX)],(r,r))
        # resized = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        return resized

    

