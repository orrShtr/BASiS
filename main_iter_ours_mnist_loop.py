import os
import numpy as np
import pandas as pd

# pytorch imports
import torch
from torch.utils.data import DataLoader, random_split


# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from utils import CofigDataset
from spectral_analysis import linear_classifier
from utils_iter_ours import train_model, modelEvaluataion, sampleAnchors

import warnings

warnings.simplefilter("ignore", UserWarning)

# hyper parameters MNIST
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')
classNum = len(classes)

anchors_per_class = 25
anchors_num = classNum * anchors_per_class

batch_size = 512 - anchors_num
iter_num = 1000

ms = 10
ms_normal = 7
sigmaFlag = 0  # 0 = sigma per class, 1 = median

ransac_nIter = 100
ransac_tol = 0.1
ransac_nPoints = 20

# siamese features
train_features_list_pd = pd.read_pickle(
    r"C:\Users\97252\PycharmProjects\LTSNet_paper\Features\mnist_train_siamese_features_list_pd.pkl")
test_features_list_pd = pd.read_pickle(
    r"C:\Users\97252\PycharmProjects\LTSNet_paper\Features\mnist_test_siamese_features_list_pd.pkl")

train_features_list_np = np.array(train_features_list_pd.drop(['label'], axis=1))
train_features_list_torch = torch.from_numpy(train_features_list_np)
print("Train features shape:")
print(train_features_list_torch.shape)

train_labels_np = np.array(train_features_list_pd.label)
train_labels_list_torch = torch.from_numpy(train_labels_np)
print("Train labels shape:")
print(train_labels_list_torch.shape)

test_features_list_np = np.array(test_features_list_pd.drop(['label'], axis=1))
featrues_test = torch.from_numpy(test_features_list_np)
print("Test features shape:")
print(featrues_test.shape)

test_labels_np = np.array(test_features_list_pd.label)
labels_test = torch.from_numpy(test_labels_np)
print("Test labels shape:")
print(labels_test.shape)

featrues_train, featrues_val, labels_train, labels_val = train_test_split(train_features_list_torch,
                                                                          train_labels_list_torch, test_size=0.2,
                                                                          random_state=0)

nmi_array = []
acc_array = []
grassmann_array = []
orthogonality_array = []
accuracy_array = []

test_num = 50
for test_iter in range(test_num):
    print("---------------------------------------")
    print(test_iter)
    print("---------------------------------------")
    model_path = r".\results\MNIST\save_MNIST_ms10_index_"
    model_path = model_path + str(test_iter)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path + '/images')
        os.mkdir(model_path + '/linear_classifier')

    anchors_index, not_anchors_index = sampleAnchors(labels_test, anchors_per_class, classNum)
    X_anchors = featrues_test[anchors_index]
    X_not_anchors = featrues_test[not_anchors_index]
    print(X_anchors.shape)
    print(X_not_anchors.shape)

    y_anchors = labels_test[anchors_index]
    y_not_anchors = labels_test[not_anchors_index]
    print(y_anchors.shape)
    print(y_not_anchors.shape)

    ## deep
    xTrain = torch.cat((X_anchors, featrues_train), dim=0)
    yTrain = torch.cat((y_anchors, labels_train), dim=0)

    xValid = torch.cat((X_anchors, featrues_val), dim=0)
    yValid = torch.cat((y_anchors, labels_val), dim=0)

    xTest = torch.cat((X_anchors, X_not_anchors), dim=0)
    yTest = torch.cat((y_anchors, y_not_anchors), dim=0)

    print('xTrain shape:')
    print(xTrain.shape)
    print('xValid shape:')
    print(xValid.shape)
    print('xTest shape:')
    print(xTest.shape)

    model, U_sampled_base = train_model(xTrain, yTrain, xValid, yValid, classes, anchors_per_class, anchors_num,
                                        batch_size,
                                        ms, ms_normal, sigmaFlag, ransac_nPoints, ransac_nIter, ransac_tol, iter_num,
                                        model_path, device)
    torch.save(U_sampled_base, model_path + '/U_sampled_base_MNIST_LN.pt')
    torch.save(xTest, model_path + '/xTest.pt')
    torch.save(yTest, model_path + '/yTest.pt')

    # load model
    # model_path = r".\results\MNIST\save_MNIST_LN_batch_512_siamese_features_1000_iters_index_"
    # model_path = model_path + str(test_iter)
    #
    # U_sampled_base_path = model_path + '/U_sampled_base_MNIST.pt'
    # U_sampled_base = torch.load(U_sampled_base_path)
    #
    # nodes_num, node_dim = xTest.shape
    # n_features = node_dim
    # projection_dim = classNum - 1
    # model = EvModule(n_features, projection_dim)
    #
    # model_fp = model_path + '/checkpoint_1000.tar'
    # model.load_state_dict(torch.load(model_fp, map_location=device.type))
    # model = model.to(device)

    nmi, acc, grassmann, orthogonality = modelEvaluataion(model, xTest, yTest, U_sampled_base, classes, ms,
                                                          ms_normal, sigmaFlag, anchors_num,
                                                          ransac_nPoints, ransac_nIter, ransac_tol)

    # # linear regression
    # model_path = model_path + '/linear_classifier'
    #
    # model.eval()  # put in evaluation mode
    # with torch.no_grad():
    #     encoded_xTrain = model(xTrain.float())
    #     encoded_xValid = model(xValid.float())
    #     encoded_xTest = model(xTest.float())
    #
    # print(encoded_xTrain.shape)
    # print(encoded_xValid.shape)
    # print(encoded_xTest.shape)
    #
    # # dataloaders - creating batches and shuffling the data
    # train_iter = CofigDataset(encoded_xTrain, yTrain)
    # train_loader = torch.utils.data.DataLoader(train_iter, batch_size=batch_size, shuffle=True, drop_last=True)
    # valid_iter = CofigDataset(encoded_xValid, yValid)
    # valid_loader = torch.utils.data.DataLoader(valid_iter, batch_size=batch_size, shuffle=False, drop_last=True)
    # test_iter = CofigDataset(encoded_xTest, yTest)
    # test_loader = torch.utils.data.DataLoader(test_iter, batch_size=batch_size, shuffle=False, drop_last=True)
    #
    # test_accuracy = linear_classifier(train_loader, valid_loader, test_loader, encoded_xTrain, classes, model_path,
    #                                   device)

    nmi_array.append(nmi)
    acc_array.append(acc)
    grassmann_array.append(grassmann)
    orthogonality_array.append(orthogonality)
    # accuracy_array.append(test_accuracy)

    print("---------Test Eval -----------")
    print("Test NMI: ", nmi)
    print("Test ACC: ", acc)
    print("Test Grassmann: ", grassmann)
    print("Test Orthogonality: ", orthogonality)
    # print("Test Accuracy: ", test_accuracy)


nmi_array_np = np.array(nmi_array)
acc_array_np = np.array(acc_array)
grassmann_array_np = np.array(grassmann_array)
orthogonality_array_np = np.array(orthogonality_array)
# accuracy_array_np = np.array(accuracy_array)

print("---------Test Eval ALL-----------")
print("NMI Mean : ", np.mean(nmi_array_np))
print("NMI STD. : ", np.std(nmi_array_np))

print("ACC Mean : ", np.mean(acc_array_np))
print("ACC STD. : ", np.std(acc_array_np))

print("Grassmann Mean : ", np.mean(grassmann_array_np))
print("Grassmann STD. : ", np.std(grassmann_array_np))

print("Orthogonality Mean : ", np.mean(orthogonality_array_np))
print("Orthogonality STD. : ", np.std(orthogonality_array_np))

# print("Accuracy Mean: ", np.mean(accuracy_array_np))
# print("Accuracy STD. : ", np.std(accuracy_array_np))
