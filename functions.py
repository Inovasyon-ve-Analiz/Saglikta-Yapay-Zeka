import torch
from torch import nn 
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

import time
import os
import sys

from dataset import CTDataset


def train(model, set, optimizer, criterion, r):
    model.train()
    size = len(set.dataset)

    for batch_idx, (X,y) in enumerate(set):
        y = torch.tensor(y).cuda().long()
        X = torch.reshape(torch.tensor(X),[len(y),1,r,r]).cuda().float()
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"loss: {loss.item():>7f}\t [{batch_idx*len(X):>5d}/{size:>5d}]")
    

def test(model, set, criterion,mode, r):
    model.eval()
    size = len(set.dataset)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for batch_idx, (X,y) in enumerate(set):
            y = torch.tensor(y).cuda().long()
            X = torch.reshape(torch.tensor(X),[len(y),1,r,r]).cuda().float()

            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= size
    print(f"{mode}: \nAccuracy: {100*correct/size:>0.1f}%, Avg Loss: {test_loss:>8f}, Correct: {correct}\n")
    return 100*correct/size


def run(model_names,lrs,wds,batch_sizes,ts,iterations,ks,rs,epochs,transfer_learning,dataset_ratio,inmeyok_range,iskemi_range,kanama_range,optimizer_name,path):
    
    model_names = [model_names] if not type(model_names) == list else model_names
    lrs = [lrs] if not type(lrs) == list else lrs
    wds = [wds] if not type(wds) == list else wds
    batch_sizes = [batch_sizes] if not type(batch_sizes) == list else batch_sizes
    ts = [ts] if not type(ts) == list else ts
    iterations = [iterations] if not type(iterations) == list else iterations
    ks = [ks] if not type(ks) == list else ks
    rs = [rs] if not type(rs) == list else rs

    train_range1 = [inmeyok_range[0],int((inmeyok_range[1]-inmeyok_range[0])*dataset_ratio)]
    train_range2 = [iskemi_range[0],int((iskemi_range[1]-iskemi_range[0])*dataset_ratio)]
    train_range3 = [kanama_range[0],int((iskemi_range[1]-kanama_range[0])*dataset_ratio)]
    test_range1 = [int((inmeyok_range[1]-inmeyok_range[0])*dataset_ratio),inmeyok_range[1]]
    test_range2 = [int((iskemi_range[1]-iskemi_range[0])*dataset_ratio),iskemi_range[1]]
    test_range3 = [int((kanama_range[1]-kanama_range[0])*dataset_ratio),kanama_range[1]]
    
    try: 
        os.mkdir("runs")
        run_number = 0
    except FileExistsError:
        run_number = len(os.listdir("runs"))

    for model_name in model_names:
        for lr in lrs:
            for wd in wds:
                for batch_size in batch_sizes:
                    for t in ts:
                        for i in iterations:
                            for k in ks:
                                for r in rs:

                                    if model_name == "resnet18":
                                        net = models.resnet18(pretrained=transfer_learning)
                                        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                                        net.fc = nn.Linear(in_features=512, out_features=3, bias=True)
                                        
                                    elif model_name == "resnet34":
                                        net = models.resnet34(pretrained=transfer_learning)
                                        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                                        net.fc = nn.Linear(in_features=512, out_features=3, bias=True)
                                        
                                    elif model_name == "resnet50":
                                        net = models.resnet50(pretrained=transfer_learning)
                                        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                                        net.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
                                        
                                    elif model_name == "resnet101":
                                        net = models.resnet101(pretrained=transfer_learning)
                                        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                                        net.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
                                        
                                    elif model_name == "resnet152":
                                        net = models.resnet152(pretrained=transfer_learning)
                                        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                                        net.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
                                        
                                    net.cuda()

                                    if optimizer_name == "Adam":
                                        optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=wd)

                                    elif optimizer_name == "SGD":
                                        optimizer = optim.SGD(net.parameters(),lr=lr,weight_decay=wd)

                                    elif optimizer_name == "RMSprop":
                                        optimizer = optim.RMSprop(net.parameters(),lr=lr,weight_decay=wd)


                                    criterion = nn.CrossEntropyLoss()

                                    preprocessing_params = [t, i, k, 0, 0, 500, 500, r]  #t,i,k,x,y,w,h,r

                                    train_data = CTDataset("labels.csv", path, preprocessing_params, train_range1,train_range2,train_range3)
                                    test_data = CTDataset("labels.csv", path, preprocessing_params, test_range1,test_range2,test_range3)
                                    
                                    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
                                    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

                                    run_number += 1
                                    dir = "runs/run" + str(run_number) 
                                    os.mkdir(dir)
                                    f = open(dir+"/results.txt","w")
                                    f.write(f"model: {model_name}\nlr: {lr}\nwd: {wd}\nbatch size: {batch_size}\npreprocessing: {preprocessing_params}\ntransfer_learning: {transfer_learning}\ntrain test ratio: {dataset_ratio}\ntrain size: {len(train_loader.dataset)}\ntest size: {len(test_loader.dataset)}\noptimizer: {optimizer_name}\nloss function: CrossEntropyLoss\nnumber of epochs: {epochs}\n\n")
                                    print(f"model: {model_name}\nlr: {lr}\nwd: {wd}\nbatch size: {batch_size}\npreprocessing: {preprocessing_params}\ntransfer_learning: {transfer_learning}\ntrain test ratio: {dataset_ratio}\ntrain size: {len(train_loader.dataset)}\ntest size: {len(test_loader.dataset)}\noptimizer: {optimizer_name}\nloss function: CrossEntropyLoss\nnumber of epochs: {epochs}")
                                    train_accuracies = []
                                    test_accuracies = []
                                    tac = 0                                    
                                    try:
                                        for e in range(epochs):
                                            tic = time.time()
                                            print(f"Epoch {e+1}\n-----------------------")
                                            train(net, train_loader, optimizer, criterion,preprocessing_params[-1])
                                            train_accuracy = test(net,train_loader,criterion,"Train",preprocessing_params[-1])
                                            test_accuracy = test(net,test_loader,criterion,"Test",preprocessing_params[-1])
                                            train_accuracies.append(train_accuracy)
                                            test_accuracies.append(test_accuracy)
                                            tac = (time.time()-tic)/60
                                            print(f"{tac} dk")

                                        f.write(f"train accuracies: {train_accuracies}\ntest accuracies: {test_accuracies}\nt: {tac}")
                                        f.close()
                                        plt.figure(run_number)
                                        plt.plot(range(len(train_accuracies)),train_accuracies)
                                        plt.plot(range(len(test_accuracies)),test_accuracies)
                                        plt.savefig(dir+"/results.png")

                                    except KeyboardInterrupt:
                                        f.write(f"train accuracies: {train_accuracies}\ntest accuracies: {test_accuracies}\nt: {tac}\nkeyboardinterrupt")
                                        f.close()
                                        plt.figure(run_number)
                                        plt.plot(range(len(train_accuracies)),train_accuracies)
                                        plt.plot(range(len(test_accuracies)),test_accuracies)
                                        plt.savefig(dir+"/results.png")
                                        sys.exit()

                                        