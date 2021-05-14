import cv2
import numpy as np

import os

for i,f in enumerate(os.listdir("crop")):
    img = cv2.imread("crop\\"+f)
    img = cv2.resize(img,(300,300))
    cv2.imwrite("resize\\"+f,img)
    print(i)

