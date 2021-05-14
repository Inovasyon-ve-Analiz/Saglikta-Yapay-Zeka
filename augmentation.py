import cv2
import pandas as pd

import os

labels = {}

for i, filename in enumerate(os.listdir("resize_train")):
    if filename=="KANAMA700.png":
        continue
    print(i)
    img = cv2.imread("resize_train\\"+filename,0)
    img2 = cv2.flip(img, 0)
    img3 = cv2.flip(img, 1)
    img4 = cv2.flip(img, -1)

    cv2.imwrite("augmented_train\\"+filename,img)
    cv2.imwrite("augmented_train\\"+filename[:-4]+"w.png",img2)
    cv2.imwrite("augmented_train\\"+filename[:-4]+"h.png",img3)
    cv2.imwrite("augmented_train\\"+filename[:-4]+"r.png",img4)

    if filename[:2] == "IN":
        labels[filename] = 0
        labels[filename[:-4]+"w.png"] = 0
        labels[filename[:-4]+"h.png"] = 0
        labels[filename[:-4]+"r.png"] = 0

    elif filename[:2] == "IS":
        labels[filename] = 1
        labels[filename[:-4]+"w.png"] = 1
        labels[filename[:-4]+"h.png"] = 1
        labels[filename[:-4]+"r.png"] = 1

    elif filename[:2] == "KA":
        labels[filename] = 2
        labels[filename[:-4]+"w.png"] = 2
        labels[filename[:-4]+"h.png"] = 2
        labels[filename[:-4]+"r.png"] = 2

   

pd.DataFrame(list(labels.items()),columns = ['column1','column2']).to_csv("labels_augmented_train.csv",index=False, header=False)

