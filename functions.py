import torch

def train(model, set, optimizer,scheduler, criterion):
    model.train()
    size = len(set.dataset)
    test_loss, correct = 0, 0

    for batch_idx, (X,y) in enumerate(set):
        y = torch.tensor(y).cuda().long()
        X = torch.reshape(torch.tensor(X),[5,1,300,300]).cuda().float()
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        test_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch_idx % 100 == 0:
            print(f"loss: {loss.item():>7f}\t [{batch_idx*len(X):>5d}/{size:>5d}]")
    
    test_loss /= size
    print(f"Train Error: \nAccuracy: {100*correct/size:>0.1f}%, Avg Loss: {test_loss:>8f}, Correct: {correct}\n")

def test(model, set, criterion,mode):
    model.eval()
    size = len(set.dataset)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for batch_idx, (X,y) in enumerate(set):
            y = torch.tensor(y).cuda().long()
            X = torch.reshape(torch.tensor(X),[5,1,300,300]).cuda().float()

            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= size
    print(f"{mode} Error: \nAccuracy: {100*correct/size:>0.1f}%, Avg Loss: {test_loss:>8f}, Correct: {correct}\n")