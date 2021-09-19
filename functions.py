import torch
from torch import nn 
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from dataset import CTDataset
import os
import time
import numpy as np


def train(model, set, optimizer, criterion, save_path, epoch):
    model.train()
    size = len(set.dataset)
    for batch_idx, (X,y) in enumerate(set):
        y = np.array(y, dtype = int)
        y = torch.tensor(y).cuda().long()
        X = torch.reshape(X, [len(y), 1, 512, 512]).cuda().float()
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"loss: {loss.item():>7f}\t [{batch_idx*len(X):>5d}/{size:>5d}]")
    

def test(model, set, criterion, mode):
    model.eval()
    size = len(set.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch_idx, (X,y) in enumerate(set):
            y = np.array(y, dtype=int)
            y = torch.tensor(y).cuda().long()
            X = torch.reshape(X, [len(y), 1, 512, 512]).cuda().float()
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size

    print(f"{mode}: \nAccuracy: {100*correct/size:>0.4f}%, Avg Loss: {test_loss:>8f}, Correct: {correct}\n")
    
    return 100*correct/size


def run(lr, wd, number_of_epoch, train_dir, test_dir, save_path, binary_classification=False, batch_size=4, checkpoint=10, model_dir=None):

    os.makedirs(save_path, exist_ok=True)
    out_features = 2 if binary_classification else 3
    net = models.resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(in_features=512, out_features=out_features, bias=True)
    
    start_epoch = 0
    if not model_dir == None:
        net.load_state_dict(torch.load(model_dir))
        start_epoch = int(model_dir.split("/")[-1].split(".")[0].split("_")[0])
        print(model_dir)

    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=wd)

    criterion = nn.CrossEntropyLoss()

    train_data = CTDataset(train_dir, binary_classification)
    test_data = CTDataset(test_dir, binary_classification)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    best_acc = 0
    for e in range(1 + start_epoch, number_of_epoch + start_epoch + 1):
        tic = time.time()
        print(f"Epoch {e}\n-----------------------")
        train(net, train_loader, optimizer, criterion, save_path, e)
        train_acc = test(net, train_loader, criterion, "Train")
        test_acc = test(net, test_loader, criterion, "Test")
       
        if test_acc > best_acc:
            print("Saving best model...")
            best_acc = test_acc
            model_path = os.path.join(save_path, str(e) + '_best.pth')
            torch.save(net.state_dict(), model_path)
                    
        if e % checkpoint == 0:
            print("Saving model...")
            model_path = os.path.join(save_path, str(e)+ '.pth')
            torch.save(net.state_dict(), model_path)
            
        print("Best Accuracy: ", best_acc)
            
        tac = (time.time()-tic)/60
        print(f"\n{tac} dk")
        
