import pydicom
from pydicom import dcmread
from matplotlib import pyplot as plt
import cv2

path = "dataset\\TRAINING\\ISKEMI\\DICOM\\10060.dcm"
max = 0
min = 0
x = dcmread(path)
print(x)
print(x.pixel_array)
for i in x.pixel_array:
    for j in i:
        if j>max: max=j
        if j<min: min=j
        
print(min)
print(max)


plt.figure(1)
plt.imshow(x.pixel_array,cmap=plt.cm.gray)


plt.figure(2)
plt.imshow(x.pixel_array>1400,cmap=plt.cm.gray)
plt.show()
