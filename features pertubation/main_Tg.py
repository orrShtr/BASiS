import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

# pytorch imports
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


from utils_Tg import sampleAnchors, ev_calculation, train_model

import warnings
warnings.simplefilter("ignore", UserWarning)

#hyper parameters MNIST
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
cmap_name = 'my_cmap'

cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

classes = ('4', '7', '9')
classNum = len(classes)

anchors_per_class = 25
anchors_num = classNum * anchors_per_class

batch_size = 512 - anchors_num
iter_num = 1500


ms = 50
ms_normal = 7
sigmaFlag = 0 # 0 = sigma per class, 1 = median

ransac_nIter = 100
ransac_tol = 0.1
ransac_nPoints = 20

if not os.path.exists('./results'):
    os.mkdir('./results')
    os.mkdir('./results/MNIST')
    
model_path = r".\results\MNIST\save_MNIST_globalT"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    os.mkdir(model_path + '/images')
    os.mkdir(model_path + '/save_features_model')
    os.mkdir(model_path + '/save_ev_model')


data_dir = 'dataset'
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
idx0 = train_dataset.targets == 4
idx1 = train_dataset.targets == 7
idx = torch.logical_or(idx0, idx1)
idx2 = train_dataset.targets == 9
idx = torch.logical_or(idx, idx2)
train_dataset.targets = train_dataset.targets[idx]
train_dataset.data = train_dataset.data[idx]

test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
idx0 = test_dataset.targets == 4
idx1 = test_dataset.targets == 7
idx = torch.logical_or(idx0, idx1)
idx2 = test_dataset.targets == 9
idx = torch.logical_or(idx, idx2)
test_dataset.targets = test_dataset.targets[idx]
test_dataset.data = test_dataset.data[idx]

train_transform = transforms.Compose([
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Set the train transform
train_dataset.transform = train_transform
# Set the test transform
test_dataset.transform = test_transform

val_size = int(len(train_dataset) * 0.2)
train_size = len(train_dataset) - int(len(train_dataset) * 0.2)
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=val_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

train_iter = iter(train_loader)
images_train, labels_train = train_iter.next()

valid_iter = iter(valid_loader)
images_valid, labels_valid = valid_iter.next()

test_iter = iter(test_loader)
images_test, labels_test = test_iter.next()

anchors_index, not_anchors_index = sampleAnchors(labels_test, anchors_per_class, classes)

X_anchors = images_test[anchors_index]
X_not_anchors = images_test[not_anchors_index]
print(X_anchors.shape)
print(X_not_anchors.shape)

y_anchors = labels_test[anchors_index]
y_not_anchors = labels_test[not_anchors_index]
print(y_anchors.shape)
print(y_not_anchors.shape)

xTrain = torch.cat((X_anchors, images_train), dim=0)
yTrain = torch.cat((y_anchors, labels_train), dim=0)

xValid = torch.cat((X_anchors, images_valid), dim=0)
yValid = torch.cat((y_anchors, labels_valid), dim=0)

xTest = torch.cat((X_anchors, X_not_anchors), dim=0)
yTest = torch.cat((y_anchors, y_not_anchors), dim=0)

nodes_num = len(xTrain)
nodes_indx_list = range(0, nodes_num)

anchors_idx = range(0, anchors_num)
sampled_nodes_indx = random.sample(nodes_indx_list, batch_size)
sampled_nodes_indx = sorted(
        sampled_nodes_indx + [indx for indx in anchors_idx if indx not in sampled_nodes_indx])

sampled_nodes_indx_len = len(sampled_nodes_indx)


inputs = xTrain[sampled_nodes_indx]
labels = yTrain[sampled_nodes_indx]
inputs_pixels = inputs.view(sampled_nodes_indx_len, -1)
U_pixels = ev_calculation(inputs_pixels, classNum, ms, ms_normal, sigmaFlag)


labels_to_plt = labels.clone()
labels_to_plt[labels == 4] = 0
labels_to_plt[labels == 7] = 1
labels_to_plt[labels == 9] = 2

train_model(xTrain, yTrain, xValid, yValid, classes, anchors_per_class, anchors_num, batch_size, ms,
                ms_normal, sigmaFlag, ransac_nPoints, ransac_nIter, ransac_tol, iter_num, model_path, device)