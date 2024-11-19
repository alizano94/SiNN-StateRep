import os
import torch
import shutil
import numpy as np
from torch import nn
from umap import UMAP
from sklearn import cluster
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CustomLoss(nn.Module):
    def __init__(self,build):
        super(CustomLoss, self).__init__()
        # perform building here

        self.loss_dict = {}
        for key in build.keys():
            self.loss_dict[key] = {
                "weight":build[key]["weight"],
                "fn":losses[key](**build[key]["params"])
            }

    def forward(self,output, target):
        # compute loss
        loss = 0
        for key in self.loss_dict.keys():
            partial_loss = self.loss_dict[key]["fn"](output, target)
            loss += self.loss_dict[key]["weight"]*partial_loss
        return loss

cluster_algorithms = {
    'kmeans':cluster.KMeans,
    'dbscan':cluster.DBSCAN,
    'hdbscan':cluster.HDBSCAN
}

dr_algorithms = {
    'pca':PCA,
    'lda':LinearDiscriminantAnalysis,
    'isomap':Isomap,
    'tsne':TSNE,
    'umap':UMAP
}

layers = {
    "Conv2d":nn.Conv2d,
    "ReLU":nn.ReLU,
    "MaxPooling2D":nn.MaxPool2d,
    "Linear":nn.Linear,
    "Flatten":nn.Flatten,
    "Unflatten":nn.Unflatten,
    "ConvTranspose2d":nn.ConvTranspose2d,
    "BatchNorm2d":nn.BatchNorm2d,
    "Upsample":nn.Upsample,
    "Sigmoid":nn.Sigmoid,
    "Dropout":nn.Dropout,
    "SoftMax":nn.Softmax,
    "LayerNorm":nn.LayerNorm,
    "TanH":nn.Tanh
}

losses = {
    "Custom":CustomLoss,
    "MSE":torch.nn.MSELoss,
    "SmoothL1":torch.nn.SmoothL1Loss,
    "CE":torch.nn.CrossEntropyLoss,
    "BCE":torch.nn.BCELoss,
    "Triplet":torch.nn.TripletMarginLoss,
    "TripletDistance":torch.nn.TripletMarginWithDistanceLoss
}

optims = {
    "Adam":torch.optim.Adam,
    "AdamW":torch.optim.AdamW,
    "SGD":torch.optim.SGD
}

def CosineDistance(x1,x2):
    cos = torch.nn.CosineSimilarity()
    return 1 - cos(x1,x2)

distances = {
    "cosine":CosineDistance,
    "euclidean":torch.nn.PairwiseDistance(p=2),
    "manhattan":torch.nn.PairwiseDistance(p=1),
}

class CustomImgClassDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)

def save_model(model,PATH=os.getcwd(),name="model.pt"):
    PATH = os.path.join(PATH,name)
    torch.save(model.state_dict(), PATH)
    print("Succesfully saved model at: \n {0}".format(PATH))

def load_model(model,PATH=os.getcwd(),name="model.pt",map_location='cpu'):
    PATH = os.path.join(PATH,name)
    model.load_state_dict(torch.load(PATH,map_location=torch.device(map_location)))
    model.eval()
    print("Succesfully loaded model from: \n {0}".format(PATH))
    print("Model architecture: \n {0}".format(model))
    return model

class OverFitStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class CovergenceStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.memory = []
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def copy_last_n_lines(source_file, destination_file, n,verbose=0):
    try:
        with open(source_file, 'r') as src:
            lines = src.readlines()[-n:]  # Get the last n lines

        with open(destination_file, 'w') as dest:
            dest.writelines(lines)

        if verbose > 0:
            print(f"Last {n} lines copied from '{source_file}' to '{destination_file}'.")
    except FileNotFoundError:
        if verbose > 0:
            print(f"File '{source_file}' not found.")
            
def centroid_based_label(data,metric,centroid_amount = 1,ignore_label=None):
    
    distances = pairwise_distances(X=data['embeddings'],Y=data['embeddings'][data['centroids']],metric=metric)
    cluster_distances = np.reshape(distances, (distances.shape[0],distances.shape[1]//centroid_amount, centroid_amount))
    if ignore_label:
        idx = np.where(data['labels'] != ignore_label)[0]
        data['labels'][idx] = np.argmin(np.sum(cluster_distances, axis=-1), axis=1)[idx]
    else:
        data['labels'] = np.argmin(np.sum(cluster_distances, axis=-1), axis=1)
        
    
    return data

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Reward')       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    plt.savefig(filename,dpi=400)
    plt.close()