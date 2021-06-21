from functions import run

#HYPERPARAMETERS            

model_names = ["resnet18","resnet50"]       #resnet18, resnet34, resnet50, resnet101, resnet152, vgg16
lrs = 1e-4                                  #1e-3
wds = 0                                     #1e-2
                                            # [0, 0.0001, 0.0003, 0.001, 0.003, 0.01]
batch_sizes = 4                             #5
                                            # [5, 10]
is_cropping = False                         #False
ts = 5                                      #5
iterations = 1                              #1
ks = 7                                      #7
rs = 100                                    #100
epochs = 20                                 #10
transfer_learning = True                    #True, False
dataset_ratio = .8                          #.75
optimizer_name = "Adam"                     #Adam, SGD, RMSprop
aug_types = {"rotation": False, "rotation45": False, "rotation315": False, "rsna": False, "rsna_rotated": False}
binary_classification = False

run(model_names,lrs, wds, batch_sizes, is_cropping, ts, iterations, ks, rs, epochs, transfer_learning,
    dataset_ratio, optimizer_name, aug_types,binary_classification)