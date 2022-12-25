import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

# pytorch imports
import torch
import torch.nn as nn

# scikit-learn imports
from sklearn.metrics import confusion_matrix

from utils import save_model
from spectral_analysis import ev_calculation, grassmann, get_orthogonality_measure, SpectralClusteringFromEV, get_acc


def calculate_accuracy(predictions, true_labels, class_num, classes, plotConfusion=False):
    cf_matrix = confusion_matrix(true_labels, predictions)
    total_correct = np.sum(np.array(predictions) == np.array(true_labels))
    total_images_num = true_labels.size(0)
    test_accuracy = total_correct / total_images_num * 100

    if plotConfusion:
        print("test accuracy: {:.3f}%".format(test_accuracy))
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.matshow(cf_matrix, aspect='auto', cmap=plt.get_cmap('Blues'))
        plt.ylabel('Actual Category')
        plt.yticks(range(class_num), classes)
        plt.xlabel('Predicted Category')
        plt.xticks(range(class_num), classes)
        for (i, j), z in np.ndenumerate(cf_matrix):
            ax.text(j, i, '{:.0f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        plt.show()
    return test_accuracy

def sampleAnchors(yTest, anchors_per_class, classNum):
    test_samples_num = len(yTest)
    indx_list = list(range(0, test_samples_num))

    for i in range(classNum):
        class_idx = torch.where(yTest == i)[0]
        idx = np.random.choice(class_idx, anchors_per_class, replace=False)
        idx = torch.from_numpy(idx)
        if i == 0:
            anchors_index = idx
        else:
            anchors_index = torch.cat((anchors_index, idx), dim=0)

    not_anchors_index = [indx for indx in indx_list if indx not in anchors_index]
    return anchors_index, not_anchors_index


def generateBase(xTrain, yTrain, nodes_indx_list, number_of_sampled_nodes, anchors_per_class, anchors_idx, classNum,
                 classes, ms, ms_normal, sigmaFlag, device):
    base_sampled_nodes_indx = random.sample(nodes_indx_list, number_of_sampled_nodes)
    base_sampled_nodes_indx = sorted(
        base_sampled_nodes_indx + [indx for indx in anchors_idx if indx not in base_sampled_nodes_indx])
    inputs = xTrain[base_sampled_nodes_indx, :]
    inputs = torch.tensor(inputs).to(device)

    labels = torch.tensor(yTrain[base_sampled_nodes_indx]).to(device)

    U_sampled_base = ev_calculation(inputs, classNum, ms, ms_normal, sigmaFlag)

    RCut_labels_sampled, model_nmi, model_acc = SpectralClusteringFromEV(U_sampled_base, labels, classNum)
    print("Base NMI : ", model_nmi)
    print("Base ACC : ", model_acc)
    return U_sampled_base


def calculateM(U_sampled_test, U_sampled):
    num_points = U_sampled_test.shape[0]

    n_digits = 4  # number of digits of precision
    U_sampled_test = torch.round(U_sampled_test * 10 ** n_digits) / (10 ** n_digits)
    U_sampled = torch.round(U_sampled * 10 ** n_digits) / (10 ** n_digits)

    pts1 = np.float32(U_sampled_test)  # from
    pts2 = np.float32(U_sampled)  # to

    A = np.concatenate([pts1, np.ones([len(pts1), 1])], axis=1)

    M, _, _, _ = np.linalg.lstsq(A, pts2, rcond=None)
    M = M.T

    return M


def AffineU(pts1, pts2, anchors_num, nPoints, nIter, tol, plotTransforms=False, saveImages=False,
            model_path=None, test_num=None):
    # Rotation+translation
    # pts1 - from (to be affined)
    # pts2 - to (base)

    N = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([N, 1])], axis=1)

    best_inliers_n = 0
    best_inliers = []

    for iter in range(nIter):
        rand_idxs = np.random.choice(np.arange(anchors_num), nPoints, replace=False)
        chosen_p1 = pts1[rand_idxs, :]
        chosen_p2 = pts2[rand_idxs, :]
        chosen_M = calculateM(chosen_p1, chosen_p2)
        chosen_U = chosen_M.dot(hpts1.T)
        chosen_U = torch.tensor(chosen_U.T)

        pts1_anchor = pts1[0:anchors_num]
        pts2_anchor = pts2[0:anchors_num]
        chosen_U_anchor = chosen_U[0:anchors_num]

        L2dists = torch.sqrt(torch.sum((chosen_U_anchor[0:anchors_num] - pts2_anchor[0:anchors_num]) ** 2, 1))
        # print(torch.mean(L2dists))
        inliers = (pts1_anchor[L2dists < tol, :], pts2_anchor[L2dists < tol, :])
        n_inliers = torch.sum(L2dists < tol)
        if n_inliers > best_inliers_n:
            best_inliers_n = n_inliers
            best_inliers = inliers

    bestM = calculateM(best_inliers[0], best_inliers[1])
    if plotTransforms:
        cmap = plt.get_cmap('GnBu')
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.0, b=0.8),
            cmap(np.linspace(0.0, 0.7, 100)))

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.matshow(bestM, cmap=new_cmap)
        for (i, j), z in np.ndenumerate(bestM):
            ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center', weight='bold')
        plt.grid(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        if saveImages:
            savefig_path = model_path + "/tramsform_test_num_" + str(test_num) + ".png"
            plt.savefig(savefig_path)
        plt.show()

    bestU = bestM.dot(hpts1.T)
    bestU = bestU.T
    return torch.tensor(bestU, dtype=float)


class EvModule(nn.Module):
    def __init__(self, n_features=2, projection_dim=1):
        super(EvModule, self).__init__()

        self.n_features = n_features
        self.projection_dim = projection_dim

        self.model = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.projection_dim),
        )

    def forward(self, x):
        return self.model(x)


def train_model(xTrain, yTrain, xValid, yValid, classes, anchors_per_class, anchors_num, number_of_sampled_nodes, ms,
                ms_normal, sigmaFlag, ransac_nPoints, ransac_nIter, ransac_tol, iter_num, model_path, device):
    nodes_num, node_dim = xTrain.shape
    classNum = len(classes)

    learning_rate = 1e-4

    # build our model and send it to the device
    n_features = node_dim
    # projection_dim = classNum - 1
    projection_dim = classNum

    # create model
    model = EvModule(n_features, projection_dim)

    # loss criterion
    criterion = nn.MSELoss()

    # optimizer - SGD, Adam, RMSProp...
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # sampling parameters
    nodes_indx_list = range(0, nodes_num)

    # base
    anchors_idx = range(0, anchors_num)
    U_sampled_base = generateBase(xTrain, yTrain, nodes_indx_list, number_of_sampled_nodes, anchors_per_class,
                                  anchors_idx, classNum, classes, ms, ms_normal, sigmaFlag, device)

    for iter in range(1, iter_num + 1):
        model.train()  # put in training mode

        sampled_nodes_indx = random.sample(nodes_indx_list, number_of_sampled_nodes)
        sampled_nodes_indx = sorted(
            sampled_nodes_indx + [indx for indx in anchors_idx if indx not in sampled_nodes_indx])

        inputs = xTrain[sampled_nodes_indx, :]
        inputs = torch.tensor(inputs).to(device)

        U_sampled = ev_calculation(inputs, classNum, ms, ms_normal, sigmaFlag)
        U_sampled = AffineU(U_sampled, U_sampled_base, anchors_num, ransac_nPoints, ransac_nIter, ransac_tol)

        # forward + backward + optimize
        outputs = model(inputs.float())  # forward pass
        loss = criterion(outputs, U_sampled.float())  # calculate the loss

        # always the same 3 steps
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backpropagation
        optimizer.step()  # update parameters


    save_model(model_path, iter, model)
    return model, U_sampled_base


def modelEvaluataion(model, xTest, yTest, U_sampled_base, classes, ms, ms_normal, sigmaFlag, anchors_num,
                     ransac_nPoints, ransac_nIter, ransac_tol):
    model.eval()

    classNum = len(classes)

    # All test set
    inputs = torch.tensor(xTest)
    labels = torch.tensor(yTest)
    with torch.no_grad():
        outputs = model(inputs.float())

    RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(outputs.detach(), labels, classNum)

    U_test = ev_calculation(inputs, classNum, ms, ms_normal, sigmaFlag)
    U_test = AffineU(U_test, U_sampled_base, anchors_num, ransac_nPoints, ransac_nIter, ransac_tol)
    model_grassmann = grassmann(outputs.detach(), U_test)

    model_orthogonality = get_orthogonality_measure(outputs, classNum)

    return model_nmi, model_acc, model_grassmann, model_orthogonality

