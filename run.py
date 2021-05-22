from functions import run

#HYPERPARAMETERS            

model_names = "resnet18"    #resnet18, resnet34, resnet50, resnet101, resnet152
lrs = 1e-4                  #1e-3
wds = [1e-2,1e-3]           #1e-2
batch_sizes = 5             #5
ts = 5                      #5
iterations = 1              #1
ks = 7                      #7
rs = 100                    #100
epochs = 5                  #10
transfer_learning = True    #True, False
dataset_ratio = .7          #.75
inmeyok_range = [0,4426]    #[0,4426]
iskemi_range = [0,1129]     #[0,1129]
kanama_range = [0,1093]     #[0,1093]
optimizer_name = "Adam"     #Adam, SGD, RMSprop
path = "../TRAINING"

run(model_names,lrs,wds,batch_sizes,ts,iterations,ks,rs,epochs,transfer_learning,
    dataset_ratio,inmeyok_range,iskemi_range,kanama_range,optimizer_name, path)