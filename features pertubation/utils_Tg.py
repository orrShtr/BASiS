import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import random


# pytorch imports
import torch
import torch.nn as nn

# scikit-learn imports
from sklearn.manifold import TSNE

from spectral_analysis import createAffinity, grassmann, get_orthogonality_measure, SpectralClusteringFromEV
from utils import save_model



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
            # nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

def sampleAnchors(yTest, anchors_per_class, classes):
    test_samples_num = len(yTest)
    indx_list = list(range(0, test_samples_num))
    classNum = len(classes)
    for i in range(classNum):
        curr_class = int(classes[i])
        class_idx = torch.where(yTest == curr_class)[0]
        idx = np.random.choice(class_idx, anchors_per_class, replace=False)
        idx = torch.from_numpy(idx)
        if i == 0:
            anchors_index = idx
        else:
            anchors_index = torch.cat((anchors_index, idx), dim=0)

    not_anchors_index = [indx for indx in indx_list if indx not in anchors_index]
    return anchors_index, not_anchors_index


def ev_calculation(features, classNum, ms, ms_normal, sigmaFlag):
    n = features.size(0)

    W = createAffinity(features, ms, ms_normal, sigmaFlag)
    s0 = torch.sum(W, axis=0)

    # L_N
    D_sqrt = torch.diag(1. / torch.sqrt(s0))
    I = torch.eye(n)
    N = I - D_sqrt @ W @ D_sqrt
    S_N, U_N = torch.linalg.eig(N)  # return ev with norm 1
    S_N = torch.real(S_N)
    U_N = torch.real(U_N)
    S_N, indices = torch.sort(S_N, dim=0, descending=False, out=None)
    U_N = U_N[:, indices]
    RCut_EV = U_N[:, 1:classNum]
    # RCut_EV = U_N[:, 0:classNum]
    return RCut_EV


class SiameseNetwork(nn.Module):
    def __init__(self, latent_dim=10):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, latent_dim))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


def valid_iter(features_model, ev_model, xValid, yValid, U_sampled_base, number_of_sampled_nodes, anchors_per_class,
               anchors_idx, classes,
               ms, ms_normal, sigmaFlag, ransac_nPoints, ransac_nIter, ransac_tol, device):
    features_model.eval()  # put in evaluation mode
    ev_model.eval()

    # loss criterion
    contrastive_criterion = ContrastiveLoss()
    mse_criterion = nn.MSELoss()

    nodes_num = xValid.shape[0]
    classNum = len(classes)
    anchors_num = len(anchors_idx)

    nodes_indx_list = range(0, nodes_num)
    sampled_nodes_indx = random.sample(nodes_indx_list, number_of_sampled_nodes)
    sampled_nodes_indx = sorted(sampled_nodes_indx + [indx for indx in anchors_idx if indx not in sampled_nodes_indx])
    sampled_nodes_indx_len = len(sampled_nodes_indx)
    if sampled_nodes_indx_len % 2 != 0:
        sampled_nodes_indx = sampled_nodes_indx[:-1]
        sampled_nodes_indx_len = len(sampled_nodes_indx)
    bd2 = int(sampled_nodes_indx_len / 2)

    inputs = xValid[sampled_nodes_indx]
    inputs = inputs.to(device)

    labels = yValid[sampled_nodes_indx]
    labels = labels.type(torch.LongTensor).to(device)

    # ---------------------
    # Eval features model
    # ---------------------
    img0 = inputs[:bd2]
    img1 = inputs[bd2:]

    labels_siamese0 = labels[:bd2]
    labels_siamese1 = labels[bd2:]

    labels_siamese = labels_siamese0.numpy() == labels_siamese1.numpy()
    labels_siamese = torch.Tensor(labels_siamese.astype(float))

    img0 = img0.to(device)
    img1 = img1.to(device)
    labels_siamese = labels_siamese.to(device)

    with torch.no_grad():
        features0, features1 = features_model(img0, img1)
    contrastive_loss = contrastive_criterion(features0, features1, labels_siamese)
    running_contrastive_loss = contrastive_loss.data.item()
    # ---------------------
    # Evaluate ev model
    # ---------------------
    with torch.no_grad():
        features = features_model.forward_once(inputs)
        ev = ev_model(features)

    U_sampled = ev_calculation(features, classNum, ms, ms_normal, sigmaFlag)
    U_sampled = AffineU(U_sampled, U_sampled_base, anchors_num, ransac_nPoints, ransac_nIter, ransac_tol)

    loss = mse_criterion(ev, U_sampled.float())  # calculate the loss
    running_mse_loss = loss.data.item()

    RCut_labels_deep, model_nmi_deep, model_acc_deep = SpectralClusteringFromEV(ev,
                                                                                labels, classNum)
    # grassman
    grassman_val = grassmann(ev, U_sampled)

    orthogonality_measure = get_orthogonality_measure(ev, classNum)

    return running_contrastive_loss, running_mse_loss, model_nmi_deep, model_acc_deep, \
           grassman_val, orthogonality_measure


def train_model(xTrain, yTrain, xValid, yValid, classes, anchors_per_class, anchors_num, number_of_sampled_nodes, ms,
                ms_normal, sigmaFlag, ransac_nPoints, ransac_nIter, ransac_tol, iter_num, model_path, device):
    tsne = TSNE(n_components=2)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
    cmap_name = 'my_cmap'

    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

    yValid_labels_to_plt = yValid.clone()
    yValid_labels_to_plt[yValid == 4] = 0
    yValid_labels_to_plt[yValid == 7] = 1
    yValid_labels_to_plt[yValid == 9] = 2


    features_model_path = model_path + "\save_features_model"
    ev_model_path = model_path + "\save_ev_model"

    nodes_num = xTrain.shape[0]
    classNum = len(classes)

    # build our model and send it to the device
    features_dim = 16
    projection_dim = classNum - 1

    # create models
    features_model = SiameseNetwork(latent_dim=features_dim)
    ev_model = EvModule(n_features=features_dim, projection_dim=projection_dim)

    # loss criterion
    contrastive_criterion = ContrastiveLoss()
    mse_criterion = nn.MSELoss()

    # sampling parameters
    nodes_indx_list = range(0, nodes_num)

    # base
    anchors_idx = range(0, anchors_num)
    base_sampled_nodes_indx = random.sample(nodes_indx_list, number_of_sampled_nodes)
    base_sampled_nodes_indx = sorted(
        base_sampled_nodes_indx + [indx for indx in anchors_idx if indx not in base_sampled_nodes_indx])
    X_base = xTrain[base_sampled_nodes_indx, :]
    y_base = yTrain[base_sampled_nodes_indx]

    # base features extracion: features_model
    features_model.eval()
    with torch.no_grad():
        features_base = features_model.forward_once(X_base)
        U_sampled_base = ev_calculation(features_base, classNum, ms, ms_normal, sigmaFlag)

    base_labels_to_plt = y_base
    base_labels_to_plt[y_base == 4] = 0
    base_labels_to_plt[y_base == 7] = 1
    base_labels_to_plt[y_base == 9] = 2

    # U_sampled_base initial
    plt.figure(figsize=(17, 9))
    plt.scatter(U_sampled_base[:, 0], U_sampled_base[:, 1], c=base_labels_to_plt, cmap=cm)
    cb = plt.colorbar()
    loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
    cb.ax.tick_params(labelsize=30)
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    plt.xlabel("$u_{1}$", fontsize=40)
    plt.ylabel("$u_{2}$", fontsize=40)
    savefig_path = model_path + "/images/U_sampled_base_initial.png"
    plt.savefig(savefig_path)
    # plt.show()

    # Valid. figures
    features_model.eval()
    ev_model.eval()
    with torch.no_grad():
        features_valid = features_model.forward_once(xValid)
        ev_valid = ev_model(features_valid)

    tsne_features_valid = tsne.fit_transform(features_valid)
    plt.figure(figsize=(17, 9))
    plt.scatter(tsne_features_valid[:, 0], tsne_features_valid[:, 1], c=yValid_labels_to_plt, cmap=cm)
    cb = plt.colorbar()
    loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
    cb.ax.tick_params(labelsize=30)
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    # plt.xlabel("$x$", fontsize=40)
    # plt.ylabel("$y$", fontsize=40)
    savefig_path = model_path + "/images/features_valid_initial.png"
    plt.savefig(savefig_path)
    # plt.show()

    plt.figure(figsize=(17, 9))
    plt.scatter(ev_valid[:, 0], ev_valid[:, 1], c=yValid_labels_to_plt, cmap=cm)
    cb = plt.colorbar()
    loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
    cb.ax.tick_params(labelsize=30)
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    plt.xlabel("$u_{1}$", fontsize=40)
    plt.ylabel("$u_{2}$", fontsize=40)
    savefig_path = model_path + "/images/ev_valid_initial.png"
    plt.savefig(savefig_path)
    # plt.show()


    # optimizer - SGD, Adam, RMSProp...
    features_learning_rate = 1e-4
    ev_learning_rate = 1e-4

    features_optimizer = torch.optim.Adam(features_model.parameters(), lr=features_learning_rate)
    ev_optimizer = torch.optim.Adam(ev_model.parameters(), lr=ev_learning_rate)

    # training loop
    iter_array = []
    training_mse_loss_array = []
    training_contrastive_loss_array = []

    nmi_array_classic = []
    nmi_array_deep = []

    acc_array_classic = []
    acc_array_deep = []

    accuracy_array_classic = []
    accuracy_array_deep = []

    grassman_array = []

    orthogonality_array = []

    valid_contrastive_loss_array = []
    valid_mse_loss_array = []
    valid_nmi_array = []
    valid_acc_array = []
    valid_accuracy_array = []
    valid_grassman_array = []
    valid_orthogonality_array = []

    strat_time = time.time()
    for iter in range(1, iter_num + 1):
    # for iter in range(iter_num):
        iter_time = time.time()

        sampled_nodes_indx = random.sample(nodes_indx_list, number_of_sampled_nodes)
        sampled_nodes_indx = sorted(
            sampled_nodes_indx + [indx for indx in anchors_idx if indx not in sampled_nodes_indx])
        sampled_nodes_indx_len = len(sampled_nodes_indx)
        if sampled_nodes_indx_len % 2 != 0:
            sampled_nodes_indx = sampled_nodes_indx[:-1]
            sampled_nodes_indx_len = len(sampled_nodes_indx)
        bd2 = int(sampled_nodes_indx_len / 2)

        inputs = xTrain[sampled_nodes_indx]
        inputs = inputs.to(device)

        labels = yTrain[sampled_nodes_indx]
        labels[labels == 4] = 0
        labels[labels == 7] = 1
        labels[labels == 9] = 2
        labels = labels.type(torch.LongTensor).to(device)

        # ---------------------
        # Train ev model
        # ---------------------
        if iter % 10 == 0:
            features_model.eval()
            ev_model.train()

            with torch.no_grad():
                features_base = features_model.forward_once(X_base)
            # global transformation
            U_sampled_base_new = ev_calculation(features_base, classNum, ms, ms_normal, sigmaFlag)
            U_sampled_base = AffineU(U_sampled_base_new, U_sampled_base, anchors_num, ransac_nPoints, ransac_nIter,
                                     ransac_tol)

            with torch.no_grad():
                features = features_model.forward_once(inputs)

            # local transformation
            U_sampled = ev_calculation(features, classNum, ms, ms_normal, sigmaFlag)
            U_sampled = AffineU(U_sampled, U_sampled_base, anchors_num, ransac_nPoints, ransac_nIter, ransac_tol)

            # forward + backward + optimize
            ev = ev_model(features)
            mse_loss = mse_criterion(ev, U_sampled.float())  # calculate the loss

            # forward + backward + optimize
            ev_optimizer.zero_grad()  # zero the parameter gradients
            mse_loss.backward()  # backpropagation
            ev_optimizer.step()  # update parameters

            running_mse_loss = mse_loss.data.item()

        # ---------------------
        # Train features model
        # ---------------------
        if iter % 1 == 0:
            features_model.train()
            ev_model.eval()

            img0 = inputs[:bd2]
            img1 = inputs[bd2:]

            ys0 = labels[:bd2]
            ys1 = labels[bd2:]

            eq_labels = ys0.numpy() == ys1.numpy()
            labels_siamese = torch.Tensor(eq_labels.astype(int))

            img0 = img0.to(device)
            img1 = img1.to(device)
            labels_siamese = labels_siamese.to(device)

            features0, features1 = features_model(img0, img1)
            contrastive_loss = contrastive_criterion(features0, features1, labels_siamese)

            # forward + backward + optimize
            features_optimizer.zero_grad()  # zero the parameter gradients
            contrastive_loss.backward()  # backpropagation
            features_optimizer.step()  # update parameters

            contrastive_running_loss = contrastive_loss.data.item()

        if iter % 10 == 0:
            # iter. evaluation
            iter_array.append(iter)
            training_mse_loss_array.append(running_mse_loss)
            training_contrastive_loss_array.append(contrastive_running_loss)
            # classic
            RCut_labels_classic, model_nmi_classic, model_acc_classic = SpectralClusteringFromEV(U_sampled, labels,
                                                                                                 classNum)
            nmi_array_classic.append(model_nmi_classic)
            acc_array_classic.append(model_acc_classic)
            # deep
            RCut_labels_deep, model_nmi_deep, model_acc_deep = SpectralClusteringFromEV(ev.detach().numpy(),
                                                                                        labels, classNum)
            nmi_array_deep.append(model_nmi_deep)
            acc_array_deep.append(model_acc_deep)

            # grassman
            grassman_val = grassmann(ev.detach(), U_sampled)
            grassman_array.append(grassman_val)

            orthogonality_measure = get_orthogonality_measure(ev.detach(), classNum)
            orthogonality_array.append(orthogonality_measure)

            # validation set
            valid_contrastive_loss, valid_mse_loss, valid_nmi, valid_acc, \
            valid_grassman, valid_orthogonality = valid_iter(features_model, ev_model, xValid, yValid, U_sampled_base,
                                                             number_of_sampled_nodes, anchors_per_class,
                                                             anchors_idx, classes,
                                                             ms, ms_normal, sigmaFlag, ransac_nPoints, ransac_nIter,
                                                             ransac_tol, device)
            valid_contrastive_loss_array.append(valid_contrastive_loss)
            valid_mse_loss_array.append(valid_mse_loss)
            valid_nmi_array.append(valid_nmi)
            valid_acc_array.append(valid_acc)
            valid_grassman_array.append(valid_grassman)
            valid_orthogonality_array.append(valid_orthogonality)

        if iter % 100 == 0:
            log = "==> Iteration: {} | Train Contrastive Loss: {:.4f} | Train MSE Loss: {:.4f} | " \
                  "| Model NMI: {:.4f} | Model ACC: {:.4f} | ".format(
                iter, contrastive_running_loss, running_mse_loss, model_nmi_deep, model_acc_deep)
            iteration_time = time.time() - iter_time
            running_time = time.time() - strat_time
            log += "Iteration Time: {:.2f} secs | ".format(iteration_time)
            log += "Running Time: {:.2f} secs".format(running_time)
            print(log)

        if iter % 10 == 0:
            # save_model(features_model_path, iter, features_model)
            # save_model(ev_model_path, iter, ev_model)

            # U_sampled_base before transformation
            plt.figure(figsize=(17, 9))
            plt.scatter(U_sampled_base_new[:, 0], U_sampled_base_new[:, 1], c=base_labels_to_plt, cmap=cm)
            cb = plt.colorbar()
            loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
            cb.ax.tick_params(labelsize=30)
            cb.set_ticks(loc)
            cb.set_ticklabels(classes)
            # plt.gca().axes.yaxis.set_ticklabels([])
            # plt.gca().axes.xaxis.set_ticklabels([])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True)
            plt.xlabel("$\~{u}_{1}$", fontsize=40)
            plt.ylabel("$\~{u}_{2}$", fontsize=40)
            savefig_path = model_path + "/images/U_sampled_base_before_Tg_iter" + str(iter) + ".png"
            plt.savefig(savefig_path)
            # plt.show()


            # U_sampled_base after transformation
            plt.figure(figsize=(17, 9))
            plt.scatter(U_sampled_base[:, 0], U_sampled_base[:, 1], c=base_labels_to_plt, cmap=cm)
            cb = plt.colorbar()
            loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
            cb.ax.tick_params(labelsize=30)
            cb.set_ticks(loc)
            cb.set_ticklabels(classes)
            # plt.gca().axes.yaxis.set_ticklabels([])
            # plt.gca().axes.xaxis.set_ticklabels([])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True)
            plt.xlabel("$u_{1}$", fontsize=40)
            plt.ylabel("$u_{2}$", fontsize=40)
            savefig_path = model_path + "/images/U_sampled_base_after_Tg_iter" + str(iter) + ".png"
            plt.savefig(savefig_path)
            # plt.show()

            # Valid. figures
            features_model.eval()
            ev_model.eval()
            with torch.no_grad():
                features_valid = features_model.forward_once(xValid)
                ev_valid = ev_model(features_valid)

            tsne_features_valid = tsne.fit_transform(features_valid)
            plt.figure(figsize=(17, 9))
            plt.scatter(tsne_features_valid[:, 0], tsne_features_valid[:, 1], c=yValid_labels_to_plt, cmap=cm)
            cb = plt.colorbar()
            loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
            cb.ax.tick_params(labelsize=30)
            cb.set_ticks(loc)
            cb.set_ticklabels(classes)
            # plt.gca().axes.yaxis.set_ticklabels([])
            # plt.gca().axes.xaxis.set_ticklabels([])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True)
            # plt.xlabel("$x$", fontsize=40)
            # plt.ylabel("$y$", fontsize=40)
            savefig_path = model_path + "/images/features_valid_iter" + str(iter) + ".png"
            plt.savefig(savefig_path)
            # plt.show()

            plt.figure(figsize=(17, 9))
            plt.scatter(ev_valid[:, 0], ev_valid[:, 1], c=yValid_labels_to_plt, cmap=cm)
            cb = plt.colorbar()
            loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
            cb.ax.tick_params(labelsize=30)
            cb.set_ticks(loc)
            cb.set_ticklabels(classes)
            # plt.gca().axes.yaxis.set_ticklabels([])
            # plt.gca().axes.xaxis.set_ticklabels([])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True)
            plt.xlabel("$u_{1}$", fontsize=40)
            plt.ylabel("$u_{2}$", fontsize=40)
            savefig_path = model_path + "/images/ev_valid_iter" + str(iter) + ".png"
            plt.savefig(savefig_path)
            # plt.show()

            plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, training_contrastive_loss_array, label='train')
    ax.plot(iter_array, valid_contrastive_loss_array, label='validation')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Contrastive Loss", fontsize=16)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    savefig_path = model_path + "/images/contrastive_loss.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, training_contrastive_loss_array, label='train')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Contrastive Loss", fontsize=16)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    savefig_path = model_path + "/images/contrastive_loss_train.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, training_mse_loss_array, label='train')
    ax.plot(iter_array, valid_mse_loss_array, label='validation')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("MSE Loss", fontsize=16)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    savefig_path = model_path + "/images/mse_loss.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, training_mse_loss_array, label='train')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("MSE Loss", fontsize=16)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    savefig_path = model_path + "/images/mse_loss_train.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, nmi_array_classic, label='classic (train)')
    ax.plot(iter_array, nmi_array_deep, label='deep (train)')
    ax.plot(iter_array, valid_nmi_array, label='deep (validation)')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("NMI", fontsize=16)
    ax.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    savefig_path = model_path + "/images/NMI.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, acc_array_classic, label='classic (train)')
    ax.plot(iter_array, acc_array_deep, label='deep (train)')
    ax.plot(iter_array, valid_acc_array, label='deep (validation)')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("ACC", fontsize=16)
    ax.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    savefig_path = model_path + "/images/ACC.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, grassman_array, label='train')
    ax.plot(iter_array, valid_grassman_array, label='validation')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Grassmann distance", fontsize=16)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    savefig_path = model_path + "/images/Grassmann.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(iter_array, orthogonality_array, label='train')
    ax.plot(iter_array, valid_orthogonality_array, label='validation')
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Orthogonality Measure", fontsize=14)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    savefig_path = model_path + "/images/Orthogonality_Measure.png"
    plt.savefig(savefig_path)
    plt.show()

    tsne = TSNE(n_components=2)

    # U_sampled_base before transformation
    plt.figure(figsize=(17, 9))
    plt.scatter(U_sampled_base_new[:, 0], U_sampled_base_new[:, 1], c=base_labels_to_plt, cmap=cm)
    cb = plt.colorbar()
    loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
    cb.ax.tick_params(labelsize=30)
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    plt.xlabel("$\~{u}_{1}$", fontsize=40)
    plt.ylabel("$\~{u}_{2}$", fontsize=40)
    savefig_path = model_path + "/images/U_sampled_base_before_Tg_iter" + str(iter) + ".png"
    plt.savefig(savefig_path)
    # plt.show()

    # U_sampled_base after transformation
    plt.figure(figsize=(17, 9))
    plt.scatter(U_sampled_base[:, 0], U_sampled_base[:, 1], c=base_labels_to_plt, cmap=cm)
    cb = plt.colorbar()
    loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
    cb.ax.tick_params(labelsize=30)
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    plt.xlabel("$u_{1}$", fontsize=40)
    plt.ylabel("$u_{2}$", fontsize=40)
    savefig_path = model_path + "/images/U_sampled_base_after_Tg_iter" + str(iter) + ".png"
    plt.savefig(savefig_path)
    # plt.show()

    features_model.eval()
    ev_model.eval()
    with torch.no_grad():
        features_valid = features_model.forward_once(xValid)
        ev_valid = ev_model(features_valid)

    tsne_features_valid = tsne.fit_transform(features_valid)
    plt.figure(figsize=(17, 9))
    plt.scatter(tsne_features_valid[:, 0], tsne_features_valid[:, 1], c=yValid_labels_to_plt, cmap=cm)
    cb = plt.colorbar()
    loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
    cb.ax.tick_params(labelsize=30)
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    # plt.xlabel("$x$", fontsize=40)
    # plt.ylabel("$y$", fontsize=40)
    savefig_path = model_path + "/images/features_valid_iter" + str(iter) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    plt.figure(figsize=(17, 9))
    plt.scatter(ev_valid[:, 0], ev_valid[:, 1], c=yValid_labels_to_plt, cmap=cm)
    cb = plt.colorbar()
    loc = np.arange(0, classNum - 1, (classNum - 1) / float(classNum)) + 0.3
    cb.ax.tick_params(labelsize=30)
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    plt.xlabel("$u_{1}$", fontsize=40)
    plt.ylabel("$u_{2}$", fontsize=40)
    savefig_path = model_path + "/images/ev_valid_iter" + str(iter) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    save_model(features_model_path, iter, features_model)
    save_model(ev_model_path, iter, ev_model)
    return features_model, ev_model, U_sampled_base
