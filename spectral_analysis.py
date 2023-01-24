import numpy as np
import matplotlib.pyplot as plt
import time
from munkres import Munkres
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import sklearn.metrics
from sklearn.metrics.cluster import normalized_mutual_info_score


# Spectral Clustering
def createAffinity(data, ms, ms_normal, sigmaFlag):
    '''
    Computes the Affinity matrix
    inputs:
    data:                   array of data featrues
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    
    returns:    
    y:                      the affinity matrix                
    '''
    n = data.shape[0]
    nbrs = NearestNeighbors(n_neighbors=ms, algorithm='kd_tree').fit(data)
    dist, idx = nbrs.kneighbors(data)
    graph_median = np.median(dist)
    dist = torch.Tensor(dist.T)
    idx = torch.Tensor(idx.T)
    id_row = torch.Tensor([range(0, n)])
    id_row = id_row.repeat(ms, 1)
    id_row = id_row.numpy()
    id_col = idx.numpy()

    if sigmaFlag == 0:
        sigma = torch.diag(1. / dist[ms_normal, :])
        W = torch.exp(-(dist @ sigma) ** 2)

    if sigmaFlag == 1:
        sigma = torch.median(dist[ms_normal, :])
        W = torch.exp(-dist ** 2 / (sigma ** 2))

    if sigmaFlag == 2:
        W = torch.exp(-dist ** 2 / (2 * graph_median ** 2))

    if sigmaFlag == 3:
        sigma = 10000
        W = torch.exp(-dist ** 2 / sigma)

    y = torch.sparse_coo_tensor([id_row.flatten(), id_col.flatten()], W.flatten(), (n, n))
    y = y.to_dense()
    y = (y + y.T) / 2
    return y



def ev_calculation(features, classNum, ms, ms_normal, sigmaFlag):
    '''
    Computes the graph Laplacian eigenvectors
    inputs:
    features:               array of data featrues
    classNum:               number of classes
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    
    returns:    
    RCut_EV:                the first K eigenvectors of L/L_N               
    '''
    n = features.size(0)

    W = createAffinity(features, ms, ms_normal, sigmaFlag)
    s0 = torch.sum(W, axis=0)

    # L
    # D = torch.diag(s0)
    # L = D - W
    # S_L, U_L = torch.linalg.eig(L) # return ev with norm 1
    # # ev_norm = torch.linalg.norm(U_L, axis=0)
    # S_L = torch.real(S_L)
    # U_L = torch.real(U_L)
    # # ev_norm1 = torch.linalg.norm(U_L, axis=0)
    # S_L, indices = torch.sort(S_L, dim=0, descending=False, out=None)
    # U_L = U_L[:, indices]
    # # RCut_EV = U_L[:, 1:classNum]
    # RCut_EV = U_L[:, 0:classNum]

    # L_N
    D_sqrt = torch.diag(1. / torch.sqrt(s0))
    I = torch.eye(n)
    N = I - D_sqrt @ W @ D_sqrt
    S_N, U_N = torch.linalg.eig(N)  # return ev with norm 1
    S_N = torch.real(S_N)
    U_N = torch.real(U_N)
    S_N, indices = torch.sort(S_N, dim=0, descending=False, out=None)
    U_N = U_N[:, indices]
    # RCut_EV = U_N[:, 1:classNum]
    RCut_EV = U_N[:, 0:classNum]
    return RCut_EV


def SpectralClusteringFromEV(ev, true_labels, classNum):
    '''
    performe spectral clutering from the spectral embedding
    inputs:
    ev:                     the eigenvectors of the graph Laplacian
    true_labels:            data true labels
    classNum:               number of classes

    returns:    
    RCut_labels:            spectral clustering assignment 
    model_nmi:              nmi value
    model_acc:              acc value
    '''
    RCut_kmeans = KMeans(n_clusters=classNum, random_state=0).fit(ev)
    RCut_labels = RCut_kmeans.labels_
    model_nmi = normalized_mutual_info_score(true_labels, RCut_labels)
    model_acc, _ = get_acc(RCut_labels, true_labels, classNum)
    return RCut_labels, model_nmi, model_acc


# Performance measures
def get_orthogonality_measure(U, classNum):
    '''
    calcute the orthogonality measure
    inputs:
    U:                      the matrix whose orthogonality is tested
    classNum:               number of classes

    returns:    
    orthogonality_measure:  orthogonality measure 
    '''
    n, m = U.shape
    ev_norm = np.linalg.norm(U, axis=0)
    ev_norm = 1 / ev_norm
    ev_norm_matrix = np.tile(ev_norm, (m, 1))
    orthogonality_matrix = U.T @ U
    orthogonality_matrix = np.multiply(np.multiply(ev_norm_matrix.T, orthogonality_matrix), ev_norm_matrix)

    dim = orthogonality_matrix.shape[0]
    I = np.eye(dim)

    orthogonality_measure = np.linalg.norm(orthogonality_matrix - I)
    return orthogonality_measure


def grassmann(A, B):
    '''
    calcute grassmann distance 
    inputs:
    A, B:                   the matrices for which the distance is checked

    returns:    
    grassmann_val:          grassmann distance between A and B 
    '''
    n, m = A.shape

    A_col_norm = torch.linalg.norm(A, dim=0)
    A_col_norm = 1 / A_col_norm
    A_norm_matrix = torch.tile(A_col_norm, (n, 1))
    A_normalized = A_norm_matrix * A  
    A_normalized = A_normalized.float()

    B_col_norm = torch.linalg.norm(B, dim=0)
    B_col_norm = 1 / B_col_norm
    B_norm_matrix = torch.tile(B_col_norm, (n, 1))
    B_normalized = B_norm_matrix * B  
    B_normalized = B_normalized.float()

    M = A_normalized.T @ B_normalized
    _, s, _ = torch.linalg.svd(M) # return ev with norm 1
    s = 1 - torch.square(s)
    grassmann_val = torch.sum(s)
    return grassmann_val


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred, confusion_matrix


def get_acc(cluster_assignments, y_true, n_clusters):
    '''
    Computes the accuracy based on the provided kmeans cluster assignments
    and true labels, using the Munkres algorithm
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    y_true = y_true.numpy()
    y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true, n_clusters)
    # calculate the accuracy
    return np.mean(y_pred == y_true), confusion_matrix


def calculate_classifier_accuracy(model, dataloader, class_num, device):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([class_num, class_num], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix



class EV_Classifier(nn.Module):
  # Linear regression model 
    def __init__(self, n_features, n_classes):
        super(EV_Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, n_classes),
        )

    def forward(self, x):
        return self.model(x)


def valid_ev_classifier(classifier_model, data_loader, criterion, device):
    classifier_model.eval()  # put in evaluation mode
    loss_array = []
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            outputs = classifier_model(inputs.float()) 
            loss = criterion(outputs, labels) 

            running_loss += loss.data.item()

    running_loss /= len(data_loader)
    return running_loss


def linear_classifier(train_loader, valid_loader, test_loader, encoded_xTrain, classes, model_path, device):
    classNum = len(classes)
    # hyper-parameters
    learning_rate = 1e-4
    epochs = 100
    classNum = 10

    n_features = encoded_xTrain.shape[1]
    classifier_model = EV_Classifier(n_features, classNum).to(device) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)

    # training loop
    running_loss_array = []
    valid_loss_array = []
    train_accuracy_array = []
    valid_accuracy_array = []
    for epoch in range(1, epochs + 1):
        classifier_model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            outputs = classifier_model(inputs.float())  
            loss = criterion(outputs, labels) 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

            running_loss += loss.data.item()

        running_loss /= len(train_loader)
        running_loss_array.append(running_loss)

        valid_loss = valid_ev_classifier(classifier_model, valid_loader, criterion, device)
        valid_loss_array.append(valid_loss)

        # Calculate training/test set accuracy of the existing model
        train_accuracy, _ = calculate_classifier_accuracy(classifier_model, train_loader, classNum, device)
        valid_accuracy, _ = calculate_classifier_accuracy(classifier_model, valid_loader, classNum, device)

        train_accuracy_array.append(train_accuracy)
        valid_accuracy_array.append(valid_accuracy)

        if epoch % 5 == 0:
            log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Valid accuracy: {:.3f}% | ".format(epoch,
                                                                                                              running_loss,
                                                                                                              train_accuracy,
                                                                                                              valid_accuracy)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

    print('==> Finished Training ...')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(running_loss_array, label='train')
    ax.plot(valid_loss_array, label='validation')
    ax.set_xlabel("epoch", fontsize=16)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=16)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    savefig_path = model_path + "/loss.png"
    plt.savefig(savefig_path)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(train_accuracy_array, label='train')
    ax.plot(valid_accuracy_array, label='validation')
    ax.set_xlabel("epoch", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    # ax.set_title("Accuracy")
    ax.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    savefig_path = model_path + "/Accuracy.png"
    plt.savefig(savefig_path)
    plt.show()

    test_accuracy, cf_matrix = calculate_classifier_accuracy(classifier_model, test_loader, classNum, device)
    print("test accuracy: {:.3f}%".format(test_accuracy))
    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.matshow(cf_matrix, aspect='auto', cmap=plt.get_cmap('Blues'))
    plt.ylabel('Actual Category')
    plt.yticks(range(classNum), classes)
    plt.xlabel('Predicted Category')
    plt.xticks(range(classNum), classes)
    for (i, j), z in np.ndenumerate(cf_matrix):
        ax.text(j, i, '{:.0f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    savefig_path = model_path + "/test_conf_mat.png"
    plt.savefig(savefig_path)
    plt.show()

    return test_accuracy