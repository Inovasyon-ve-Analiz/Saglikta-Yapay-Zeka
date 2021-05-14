import torch
from torch import nn 
from torch.nn import functional as F
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt

import time
import sys

from dataset import CTDataset, DCMDataset
from functions import train, test

train_data = CTDataset("labels_augmented_train.csv","augmented_train")
test_data = CTDataset("labels_test.csv","resize_test")

train_loader = DataLoader(train_data,batch_size=5,shuffle=True)
test_loader = DataLoader(test_data,batch_size=5,shuffle=True)

"""
train_features,train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Labels: {train_labels}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

"""

net = models.resnet50(pretrained=True)
net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
net.cuda()

optimizer = optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=510,gamma=0.1)
criterion = nn.CrossEntropyLoss()

epochs = 50
for i in range(epochs):
    tic = time.time()
    print(f"Epoch {i+1}\n-----------------------")
    train(net, train_loader, optimizer,scheduler, criterion)
    test(net,test_loader,criterion,"Train")
    test(net,test_loader,criterion,"Test")
    t = (time.time()-tic)/60
    print(f"{t} dk")
print("Done")
