import os
import math
import pickle
import shutil
import random 
import hdbscan
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt # plotting library
from sklearn.metrics.pairwise import pairwise_distances



import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .misc import *
from .visualize import *
from .state_represent import *

# The ConvNet class is a neural network model that takes in input data and creates a sequential model
# with specified layers and parameters.
class ConvNet(nn.Module):
    def __init__(self, input_data, k = None):
        super().__init__()

        self.input_data = input_data

        if k == None:
            self.k = len(os.listdir(input_data['data_path']))
        else:
            self.k = k

        self.model = nn.Sequential()
        self.in_feat = 0
        for key in self.input_data['classifier'].keys():
            self.model.append(
                layers[self.input_data['classifier'][key]["type"]](
                    **self.input_data['classifier'][key]['params']
                )
            )
            if self.input_data['classifier'][key]["type"] == "Linear":
                self.in_feat =  self.input_data['classifier'][key]["params"]["out_features"]

        if self.input_data['final_layer']:
            self.model.append(layers["Linear"](**{"in_features":self.in_feat, "out_features":self.k}))
        if bool(self.input_data["verbose"]) == True:
            print(self.model)
    
    def forward_logits(self, x):
        
        return self.model(x)
    
    def forward(self,x):
        return self.model(x)

class Classify():
    def __init__(self,input_data):
        super().__init__()
        self.input_data = input_data
        self.channels = input_data['classify']['channels']
        self.image_size = input_data['classify']['image_size']
        self.bi_thr =  input_data['classify']['binary_threshold']
        self.k = input_data['classify']['states']
        self.metric = input_data['classify']['metric']

        
        # Check if the GPU is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        self.cnn = ConvNet(self.input_data['classify'],k=self.k).to(self.device)

        self.train_params = self.input_data['classify']['train_params']

        self.data_dir =self.input_data['classify']['data_path']
        
    def load_data_centroid_based(self,data,show=True,n_samples=50,n_transforms=100,batch_size=32,shuffle=True):
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ]
        )
        theta = np.linspace(start=0.0, stop=360, num=n_transforms-2)
        
        labels = np.unique(data['labels'])
        n_centroids  = len(labels)
        if -1 in labels:
            n_centroids -= 1
        
        N_data  = n_centroids*n_samples*n_transforms
        self.data_imgs = torch.zeros(N_data, self.channels, self.image_size[0], self.image_size[1])
        self.data_labels = torch.zeros(N_data,dtype=torch.long)
        # self.data_labels = np.zeros(shape=(N_data),dtype=int)
        
        ordered_idx = []
        self.img_idx = []
        
        # Order images according to centroid proximity
        for l in range(n_centroids):
            idx = np.where(data['labels'] == l)[0] # retrieves images inside a cluster
            distances = pairwise_distances(X=data['embeddings'][idx],Y=data['embeddings'][data['centroids'][l]].reshape(1, -1),metric=self.metric) # gets distances from centroid
            ordered_idx.append(idx[np.argsort(distances.flatten())].tolist())
            
        # select images based on the distance to centroid    
        for c in range(n_centroids):
            for s in range(n_samples):
                self.img_idx.append(ordered_idx[c][s])
        self.img_idx = np.array(self.img_idx,dtype=int)
        
        # Retrieve the name and labels of the selected images
        self.labels = np.array(data['labels'])[self.img_idx]
        self.names = np.array(data['names'])[self.img_idx]
        
        for i in tqdm(range(len(self.names))):
            img = Image.open(os.path.join(self.data_dir,self.names[i]))
            for j in range(n_transforms):
                if j == 0:
                    img = transforms.functional.hflip(img)
                elif j == 1:
                    img = transforms.functional.vflip(img)
                else:
                    img = transforms.functional.rotate(img,angle=theta[j-2])
                self.data_imgs[i*n_transforms + j] += torch.unsqueeze(transform(img),0)[0]
                self.data_labels[i*n_transforms + j] += self.labels[i]
        
        # self.data_labels = torch.tensor(self.data_labels,requires_grad=True,dtype=torch.long)
                
        self.data_set = CustomImgClassDataset(images=self.data_imgs,labels=self.data_labels)
        
        train_data, val_data, test_data = random_split(
            self.data_set, [0.8,0.1,0.1]
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=shuffle
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=shuffle
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=shuffle
        )
        
        self.sample = next(iter(self.test_dataloader))
        self.sample = [self.sample[0][0:3],self.sample[1][0:3]]
        
    def load_data(self,data,show=True,batch_size=32,shuffle=True):
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ]
        )
        
        labels = np.unique(data['labels'])
        
        N_data  = len(np.array(data['labels']).flatten())
        self.data_imgs = torch.zeros(N_data, self.channels, self.image_size[0], self.image_size[1])
        self.data_labels = torch.zeros(N_data,dtype=torch.long)
        
        # Retrieve the name and labels of the selected images
        self.labels = np.array(data['labels'])
        self.names = np.array(data['names'])
        
        for i in tqdm(range(len(self.names))):
            img = Image.open(os.path.join(self.data_dir,self.names[i]))
            self.data_imgs[i] += torch.unsqueeze(transform(img),0)[0]
            self.data_labels[i] += self.labels[i]
                        
        self.data_set = CustomImgClassDataset(images=self.data_imgs,labels=self.data_labels)
        
        train_data, val_data, test_data = random_split(
            self.data_set, [0.8,0.1,0.1]
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=shuffle
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=shuffle
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=shuffle
        )
        
        self.sample = next(iter(self.test_dataloader))
        self.sample = [self.sample[0][0:3],self.sample[1][0:3]]
        
    def load_data_bkp(self,show=True,n_samples=3,batch_size=32,shuffle=True):
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.image_size),
                v2.RandomHorizontalFlip(p=0.85),
                v2.RandomVerticalFlip(p=0.85),
                v2.RandomRotation(degrees=270),
                transforms.ToTensor(),
            ]
        )
        data_dir = "/Users/admin/Documents/LSU/TangGroup/SAC3/data/cnn"
        dataset = datasets.ImageFolder(
            self.data_dir, 
            transform=transform
        )

        train_data, val_data, test_data = random_split(
            dataset, [0.8,0.1,0.1]
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=shuffle
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=shuffle
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=shuffle
        )

        self.sample = next(iter(self.test_dataloader))
        self.sample = [self.sample[0][0:n_samples],self.sample[1][0:n_samples]]
    
    def train(
            self,
            num_epochs = 30, 
            seed = 0,
            early_stops=[],
            save_path='./',
            chkpt_name='checkpoint',
            chkpt_type='best'
        ):
        ### Define the loss function
        self.loss_fn = losses[self.train_params['loss_fn']['loss_type']](**self.train_params['loss_fn']['loss_params'])

        ### Set the random seed for reproducible results
        torch.manual_seed(seed)

        params_to_optmize = [{'params': self.cnn.parameters()}]

        self.optimizer = optims[self.train_params['optimizer']['type']](
            params=params_to_optmize,
            **self.train_params['optimizer']['params']
        )
        
        best_loss = math.inf

        diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.test_epoch()
            for es in early_stops:
                if es.early_stop(val_loss):
                    break
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch, num_epochs,train_loss,val_loss))
            diz_loss['train_loss'].append(train_loss)
            diz_loss['val_loss'].append(val_loss)
            self.plt_cnn_output(
                save_path=save_path,
                name="sample_predcition_{0}_epoch.png".format(
                    epoch
                )
            )
            # save checkpoints
            if chkpt_type == "epoch":
                cnn_name = "cnn_{0}_{1}_epoch.pt".format(chkpt_name,epoch)
                save_model(model=self.cnn,PATH=save_path,name=cnn_name) 
            elif chkpt_type == "best" and best_loss > val_loss:
                cnn_name = "cnn_{0}_{1}_epoch.pt".format(chkpt_name,chkpt_type)
                save_model(model=self.cnn,PATH=save_path,name=cnn_name)
                best_loss = val_loss
            else: pass
            plot_loss(loss=diz_loss,path=save_path, name='cnn_learning.png')

    ### Training function
    def train_epoch(self):
        # Set train mode for both the cnn
        self.cnn.train()
        train_loss = []
        # Iterate the dataloader 
        for batch_image, batch_labels in tqdm(self.train_dataloader): 
            
            # Backward pass
            self.optimizer.zero_grad()
            
            #  Move tensor to the proper device
            batch_image = batch_image.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Predict label
            p = self.cnn(batch_image)
            
            # pred_label = torch.argmax(self.cnn(image_batch),dim=1)
            loss = self.loss_fn(p, batch_labels)
            loss.backward()
            self.optimizer.step()
            # Print batch loss
            # print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    ### Testing function
    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        self.cnn.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for batch_image, batch_labels in tqdm(self.val_dataloader):
                # Move tensor to the proper device
                batch_image = batch_image.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # Predict label
                p = self.cnn(batch_image)
                # pred_label = torch.argmax(self.cnn(image_batch),dim=1)
                # Append the network output and the original image to the lists
                conc_out.append(p.cpu())
                conc_label.append(batch_labels.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label) 
            # Evaluate global loss
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data
    
    def encode_data_set(self, data, name='embeddings.pickle',save_path=None,data_dir=None):
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ]
        )
        self.cnn.eval()
        
        if data_dir == None:
            path = self.data_dir
        else:
            path = data_dir
        names = data['names']
        embeddings = torch.zeros(len(names), self.k)
        labels = torch.zeros(len(names),dtype=int)
        print("Attempting data embedding...")
        for i in tqdm(range(len(names))):
            img = torch.unsqueeze(transform(Image.open(os.path.join(path,names[i]))),0)[0]
            img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
            embeddings[i] += self.cnn.forward(img).detach().numpy()
            labels[i] += int(torch.argmax(embeddings[i]))
        data['embeddings'] = embeddings
        data['labels'] = labels
        print("Data embedding done")
        
        print("Saving file....")
        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("""succesfully encoded images!
            Output file saved at: {0}""".format(
                save_file
            ))
        
        return data
    
    def plt_cnn_output(self,save_path='./',name=None):
        # Create figure
        n_samples = len(self.sample[1])
        fig = plt.figure(layout='constrained', figsize=(10, 4))
        subfigs = fig.subfigures(1, n_samples, wspace=0.07)
        
        # Retrieve sample images and labels
        imgs = self.sample[0]
        labels = self.sample[1]
        
        # Set CNN to evaluation mode
        self.cnn.eval()
        
        # Predict labels
        prediction = nn.functional.softmax(self.cnn(imgs.to(self.device)))
        
        
        # Begin plotting
        for i in range(n_samples):
            # Plot original image
            ax = subfigs[i].subplots(1, 1)
            ax.imshow(imgs[i].permute(1, 2, 0),cmap='gray')
            # Hide X and Y axes label marks
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Predicted label {0}, Real label {1} \n Uncertainty :{2:.2f}'.format(
                int(torch.argmax(prediction[i])),
                labels[i],
                1-prediction[i][int(torch.argmax(prediction[i]))]
                )
            )
            # Hide X and Y axes label marks
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)

        if name is not None:
                plt.savefig(
                    os.path.join(
                        save_path,
                        name
                    ),
                    dpi=400
                )
        # plt.cla()
        plt.close('all')
            # return fig
