import torch
from torch import nn 
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from dataset import CTDataset
import time
import os
import sys
import numpy as np

def augmentation(aug_types):
    img_dirs = ["TRAINING"]
    for aug in aug_types:
        if aug == "rotation" and aug_types[aug] == True:
            img_dirs.append("rotated")
        elif aug == "rotation45" and aug_types[aug] == True:
            img_dirs.append("rotated45")
        elif aug == "rotation315" and aug_types[aug] == True:
            img_dirs.append("rotated315")
    return img_dirs

def train(model, set, optimizer, criterion, r):
    model.train()
    size = len(set.dataset)
    for batch_idx, (X,y) in enumerate(set):
        y = np.array(y, dtype = int)
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
            y = np.array(y, dtype=int)
            y = torch.tensor(y).cuda().long()
            X = torch.reshape(torch.tensor(X),[len(y),1,r,r]).cuda().float()
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= size
    print(f"{mode}: \nAccuracy: {100*correct/size:>0.1f}%, Avg Loss: {test_loss:>8f}, Correct: {correct}\n")
    return 100*correct/size


def run(model_names,lrs,wds,batch_sizes,is_cropping,ts,iterations,ks,rs,number_of_epoch,transfer_learning,ratio,optimizer_name,aug_types):
    
    model_names = [model_names] if not type(model_names) == list else model_names
    lrs = [lrs] if not type(lrs) == list else lrs
    wds = [wds] if not type(wds) == list else wds
    batch_sizes = [batch_sizes] if not type(batch_sizes) == list else batch_sizes
    ts = [ts] if not type(ts) == list else ts
    iterations = [iterations] if not type(iterations) == list else iterations
    ks = [ks] if not type(ks) == list else ks
    rs = [rs] if not type(rs) == list else rs
    
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
                                    if is_cropping:
                                        preprocessing_params = [t, i, k, 0, 0, 500, 500, r]  #t,i,k,x,y,w,h,r
                                    else:
                                        preprocessing_params = [t, i, k, 0, 0, 500, 500, 512]  #t,i,k,x,y,w,h,

                                    img_dirs= augmentation(aug_types)

                                    train_data = CTDataset(img_dirs, preprocessing_params, is_cropping, 1, ratio)
                                    test_data = CTDataset(img_dirs, preprocessing_params, is_cropping, 0, ratio)

                                    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
                                    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

                                    run_number += 1
                                    dir = "runs/run" + str(run_number) 
                                    os.mkdir(dir)
                                    with open(f"{dir}/run{run_number} results.txt","w") as file:
                                        run_data = "".join([f"run number: {run_number}\nmodel: {model_name}\nlr: {lr}\nwd: {wd}\n",
                                                   f"batch size: {batch_size}\nis_cropping: {is_cropping}",
                                                   f"\npreprocessing: {preprocessing_params}\ntransfer_learning: {transfer_learning}",
                                                   f"\ntrain test ratio: {ratio}\ntrain size: {len(train_loader.dataset)}",
                                                   f"\ntest size: {len(test_loader.dataset)}\noptimizer: {optimizer_name}",
                                                   f"\nloss function: CrossEntropyLoss\nnumber of epochs: {number_of_epoch}\n\n"])

                                        print(run_data)
                                        file.write(run_data)

                                    train_accuracies = []
                                    test_accuracies = []
                                    epochs = []
                                    for e in range(1, number_of_epoch + 1):
                                        tic = time.time()
                                        print(f"Epoch {e}\n-----------------------")
                                        train(net, train_loader, optimizer, criterion,preprocessing_params[-1])
                                        train_accuracy = test(net,train_loader,criterion,"Train",preprocessing_params[-1])
                                        test_accuracy = test(net,test_loader,criterion,"Test",preprocessing_params[-1])
                                        train_accuracies.append(train_accuracy)
                                        test_accuracies.append(test_accuracy)
                                        tac = (time.time()-tic)/60
                                        print(f"{tac} dk")
                                        with open(f"{dir}/run{run_number} results.txt","w") as file:
                                            epoch = "".join([f"Epoch {e} train accuracy: {train_accuracy}\n",
                                                    f"Epoch {e} test accuracy: {test_accuracy}\nt: {tac}\n\n"])
                                            epochs.append(epoch)
                                            general_info = "".join([f"Training Accuracies:\nMax: {max(train_accuracies)}\n"
                                                                    f"Min: {min(train_accuracies)}\nMean: {np.mean(train_accuracies)}\n\n"
                                                                    f"Test Accuracies:\nMax: {max(test_accuracies)}\n"
                                                                    f"Min: {min(test_accuracies)}\nMean: {np.mean(test_accuracies)}"])
                                            file.write(run_data + "".join(epochs) + general_info)

                                        if e > 1:
                                            plt.clf()
                                            plt.figure(f"Run {run_number}")
                                            plt.plot(range(len(train_accuracies)),train_accuracies)
                                            plt.plot(range(len(test_accuracies)),test_accuracies)
                                            plt.savefig(f"{dir}/run{run_number} results ({e} epochs).png")
                                        if e > 2:
                                            os.remove(f"{dir}/run{run_number} results ({e-1} epochs).png")



                                        