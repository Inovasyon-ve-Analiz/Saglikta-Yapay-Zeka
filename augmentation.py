import cv2
import pandas as pd

import os
"""
aug_types = {"rotate":cv2.rotate}

def augmentation(aug_type, path="TRAINING"):
    img = cv2.imread("TRAINING\\INMEYOK0.png",0)
    image = aug_types[aug_type](img,cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("fasdsdgf",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

augmentation("rotate")

"""
os.chdir("..")
files = os.listdir("TRAINING")   
for i, filename in enumerate(files):
        
    img = cv2.imread("TRAINING\\"+filename,0)
    image = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("TRAINING\\"+filename,img)
    cv2.imwrite("rotated\\"+filename[:-4]+"r.png",image)

    k = cv2.waitKey(0)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
