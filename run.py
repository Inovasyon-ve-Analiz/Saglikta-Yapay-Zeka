from functions import run

#HYPERPARAMETERS            

model_names = "resnet18"    #resnet18, resnet34, resnet50, resnet101, resnet152
lrs = 1e-4                  #1e-3
wds = 0.003               #1e-2
# [0, 0.0001, 0.0003, 0.001, 0.003, 0.01]
batch_sizes = 10             #5
is_cropping = False         #False
ts = 5                      #5
iterations = 1              #1
ks = 7                      #7
rs = 100                    #100
epochs = 15                 #10
transfer_learning = True    #True, False
dataset_ratio = .8          #.75
optimizer_name = "Adam"     #Adam, SGD, RMSprop
is_rotated = True
aug_types = {"rotation": is_rotated}

run(model_names,lrs, wds, batch_sizes, is_cropping, ts, iterations, ks, rs, epochs, transfer_learning,
    dataset_ratio, optimizer_name, aug_types)