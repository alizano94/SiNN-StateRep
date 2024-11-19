import os
import math
import pickle
import shutil
import random 
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
from PIL import Image
from umap import UMAP
from tqdm import tqdm
from sklearn import cluster
import matplotlib.pyplot as plt # plotting library
from sklearn.metrics.pairwise import pairwise_distances

import pathlib

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .misc import *
from .visualize import plot_loss

# The Encoder class is a neural network module that takes input data, applies a series of
# convolutional layers followed by a fully connected layer, and produces embeddings as output.
class Encoder(nn.Module):    
    def __init__(self, input_data):
        """
        The function initializes a neural network model by creating a list of modules based on the input
        data.
        
        :param input_data: The `input_data` parameter is a dictionary that contains the following keys
        and values:
        """
        super().__init__()
        self.input_data = input_data

        self.model = nn.Sequential()
        for key in self.input_data['encoder'].keys():
            self.model.append(
                layers[self.input_data['encoder'][key]["type"]](
                    **self.input_data['encoder'][key]['params']
                )
            )
        
        
       # define fully connected layer to create embeddings
        if bool(self.input_data["add_linear"]) == True:
            self.model.append(layers["Flatten"]())
            self.model.append(
                layers["Linear"](
                    np.prod(self.input_data['shape_before_flattening']),
                    self.input_data['latent_dim']
                )
            )
        if bool(self.input_data["verbose"]) == True:
            print(self.model)
    def forward(self, x):
        """
        The forward function takes an input tensor, passes it through a series of convolutional layers,
        flattens the tensor, and passes it through a final layer before returning the output.
        
        :param x: The parameter `x` represents the input tensor that is passed through the network
        :return: the output of the last module in the neural network.
        """
        
        return self.model(x)
    
# The Decoder class is a neural network module that takes in a latent vector and generates an output
# tensor by applying a series of layers, including fully connected and convolutional layers.
class Decoder(nn.Module):    
    def __init__(self, input_data):
        """
        The function initializes a neural network model with a specified architecture for a given input
        data.
        
        :param input_data: The `input_data` parameter is a dictionary that contains the following
        information:
        """
        super().__init__()
        self.input_data = input_data

        self.model = nn.Sequential()
        self.reshape_dim = self.input_data['shape_before_flattening']

        
        # define fully connected layer to create embeddings
        if bool(self.input_data["add_linear"]) == True:
            self.model.append(
                layers["Linear"](
                    self.input_data['latent_dim'],
                    np.prod(self.reshape_dim)
                )
            )
            self.model.append(layers["Unflatten"](1, torch.Size(self.reshape_dim)))

        self.out_channels = 1
        for key in self.input_data['decoder'].keys():
            self.model.append(
                layers[self.input_data['decoder'][key]["type"]](
                    **self.input_data['decoder'][key]['params']
                )
            )
        if self.input_data['use_logits'] == True:
            self.model.append(layers["Sigmoid"]())
        else:    
            self.model.append(layers["TanH"]())
        if bool(self.input_data["verbose"]) == True:
            print(self.model)
            
    def forward(self, x):
        """
        The forward function takes an input tensor, applies a fully connected layer, reshapes the
        tensor, and then passes it through a series of convolutional layers.
        
        :param x: The parameter `x` represents the input tensor that is passed through the forward
        method of the model. It is a tensor that contains the input data for the model
        :return: the output tensor after passing it through the convolutional layers.
        """

        return self.model(x)

# The `ThresholdTransform` class takes a threshold value in the range [0, 255] and converts it to the
# range [0, 1], then applies the thresholding operation to input data.
class ThresholdTransform(object):
  def __init__(self, thr_255):
    """
    This Python function initializes an object with a threshold value converted from a range of
    [0..255] to [0..1].
    
    :param thr_255: The `thr_255` parameter represents the threshold value for gray levels in the
    range of [0, 255]. This value is then divided by 255 to convert it to the range of [0, 1], which
    is commonly used in many image processing applications
    """
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    """
    The function returns a boolean array indicating whether each element in the input array is greater
    than a threshold value.
    
    :param x: The code snippet you provided defines a `__call__` method that takes a parameter `x` and
    compares it to a threshold value `self.thr`. If `x` is greater than the threshold, it returns a
    boolean array with the same data type as `x`
    :return: The code snippet is defining a `__call__` method for a class, which takes a parameter
    `x`. Inside the method, it is comparing `x` with a threshold value `self.thr` and returning a
    boolean array where each element is `True` if the corresponding element in `x` is greater than
    `self.thr`, and `False` otherwise. The `.to(x
    """
    return (x > self.thr).to(x.dtype)  # do not change the data type

# The AE class is an implementation of an Autoencoder in Python, which includes methods for loading
# and preprocessing data, training the autoencoder, and encoding the dataset.
class AE():
    def __init__(self,input_data):
        """
        The function initializes an object with an encoder and decoder ANN, train parameters, and a data
        directory.
        
        :param input_data: The `input_data` parameter is a dictionary that contains the following keys:
        """
        super().__init__()
        self.input_data = input_data
        self.img_shape = self.input_data['represent']["image_size"]
        self.bi_thr = self.input_data['represent']["binary_threshold"]

        
        # Check if the GPU is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')
        
        # Create encoder/decoder models
        self.encoder = Encoder(self.input_data['represent'])
        self.decoder = Decoder(self.input_data['represent'])        

        self.train_params = self.input_data['represent']['train_params']

        self.data_dir =self.input_data['represent']['data_path']

    def load_data(self,batch_size=32,shuffle=True):
        """
        The `load_data` function loads and preprocesses image data, splits it into training, validation,
        and test sets, and returns a sample of the test data.
        
        :param show: The "show" parameter determines whether or not to display a sample of the data set.
        If set to True, a sample of the data set will be displayed. If set to False, no sample will be
        displayed, defaults to True (optional)
        :param n_samples: The `n_samples` parameter determines the number of samples to display from the
        test dataset, defaults to 3 (optional)
        :param batch_size: The batch size is the number of samples that will be propagated through the
        network at once. It is used to control the number of samples processed in each iteration during
        training, defaults to 32 (optional)
        :param shuffle: The "shuffle" parameter determines whether the data will be shuffled before each
        epoch during training. If set to True, the data will be randomly shuffled. If set to False, the
        data will be processed in the order it is loaded, defaults to True (optional)
        :return: If the `show` parameter is set to `True`, the function will return a
        `matplotlib.figure.Figure` object. Otherwise, it will return `None`.
        """
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.img_shape),
                transforms.ToTensor(),
            ]
        )
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

        self.sample = next(iter(self.test_dataloader))[0][0:3]
        return None

    def train(
            self,
            num_epochs = 30, 
            seed = 0,
            early_stops=[],
            save_path='./',
            name='checkpoint',
            chkpt_type='best',
            show=True
        ):
        
        """
        The `train` function trains an autoencoder model for a specified number of epochs, using a
        specified loss function and optimizer, and saves the training and validation loss values.
        
        :param num_epochs: The number of epochs is the number of times the model will iterate over the
        entire training dataset. Each epoch consists of a forward pass, a backward pass, and an update
        of the model's parameters based on the computed gradients, defaults to 30 (optional)
        :param seed: The seed parameter is used to set the random seed for reproducible results. By
        setting a specific seed value, you can ensure that the random number generation during training
        is consistent across different runs. This can be useful for debugging or comparing different
        models, defaults to 0 (optional)
        :param early_stops: The "early_stops" parameter is a list of early stopping criteria. Each early
        stopping criterion is an object that has a method called "early_stop". This method takes the
        validation loss as input and returns a boolean value indicating whether to stop training early
        based on the current validation loss. If any of
        """
        if show:
            fig = plt.figure(layout='constrained', figsize=(10, 4))
            subfigs = fig.subfigures(1, 3, wspace=0.07)
            for i in range(3):
                ax = subfigs[i].subplots(1, 1)
                ax.imshow(self.sample[i].permute(1, 2, 0),cmap='gray')
                # Hide X and Y axes label marks
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)

                # Hide X and Y axes tick marks
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle('Test sample from data set', fontsize='xx-large')
            plt.savefig(os.path.join(save_path,'data_set_sample.png'),dpi=400)
        
        ### Define the loss function
        self.loss_fn = losses[self.train_params['loss_fn']['loss_type']](**self.train_params['loss_fn']['loss_params'])
        ### Set the random seed for reproducible results
        torch.manual_seed(seed)

        params_to_optmize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

        self.optimizer = optims[self.train_params['optimizer']['type']](
            params=params_to_optmize,
            **self.train_params['optimizer']['params']
        )

        # Move both the encoder and the decoder to the selected device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        
        best_loss = math.inf

        diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.test_epoch()
            for es in early_stops:
                if es.early_stop(val_loss):
                    break
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            diz_loss['train_loss'].append(train_loss)
            diz_loss['val_loss'].append(val_loss)
            self.plot_ae_outputs(save_path=save_path,name="sample_reconstruction_{0}_epoch.png".format(epoch))
            
            # Save dictionary
            save_file = os.path.join(save_path, "autoencoder_learning.pickle")
            with open(save_file, 'wb') as handle:
                pickle.dump(diz_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # save checkpoints
            if chkpt_type == "epoch":
                enc_name = "enc_{0}_{1}_epoch.pt".format(name,epoch)
                dec_name = "dec_{0}_{1}_epoch.pt".format(name,epoch)
                save_model(model=self.encoder,PATH=save_path,name=enc_name)
                save_model(model=self.decoder,PATH=save_path,name=dec_name) 
            elif chkpt_type == "best" and best_loss > val_loss:
                enc_name = "enc_{0}_{1}_epoch.pt".format(name,chkpt_type)
                dec_name = "dec_{0}_{1}_epoch.pt".format(name,chkpt_type)
                
                save_model(model=self.encoder,PATH=save_path,name=enc_name)
                save_model(model=self.decoder,PATH=save_path,name=dec_name) 

                best_loss = val_loss
            else: pass

            plot_loss(loss=diz_loss,path=save_path, name='autoencoder_learning.png')

    ### Training function
    def train_epoch(self):
        """
        The function `train_epoch` trains an encoder-decoder model for one epoch using unsupervised
        learning.
        :return: the mean of the train loss.
        """
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in tqdm(self.train_dataloader): # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Backward pass
            self.optimizer.zero_grad()
            #  Move tensor to the proper device
            image_batch = image_batch.to(self.device)
            # Encode data
            encoded_data = self.encoder(image_batch)
            # Decode data
            decoded_data = self.decoder(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, image_batch)
            loss.backward()
            self.optimizer.step()
            # Print batch loss
            # print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    ### Testing function
    def test_epoch(self):
        """
        The function `test_epoch` evaluates the performance of the encoder-decoder network on the
        validation dataset by calculating the loss between the network's output and the original images.
        :return: the value of the variable `val_loss.data`.
        """
        # Set evaluation mode for encoder and decoder
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for image_batch, _ in tqdm(self.val_dataloader):
                # Move tensor to the proper device
                image_batch = image_batch.to(self.device)
                # Encode data
                encoded_data = self.encoder(image_batch)
                # Decode data
                decoded_data = self.decoder(encoded_data)
                # Append the network output and the original image to the lists
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label) 
            # Evaluate global loss
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data

    def label_data(self, data:dict, centroids, metric='cosine', save_path=None, name="sinn_cleaned_embeddings.pickle") -> np.ndarray:   
        
        if len(data['embeddings'].shape) >= 2:
            data['embeddings'] = data['embeddings'].reshape(-1,data['embeddings'].shape[-1])          
        # code for relabeling.
        distances : np.ndarray = pairwise_distances(X=data['embeddings'],Y=centroids,metric=metric)
        cluster_distances : np.ndarray = np.reshape(distances, (distances.shape[0],distances.shape[1]//1, 1))
        
        data['labels'] = np.argmin(np.sum(cluster_distances, axis=-1), axis=1)

        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return data
    
    def plot_ae_outputs(self, save_path='./', name=None):
        """
        The function `plot_ae_outputs` plots the original and reconstructed images from an autoencoder
        model.
        
        :param name: The `name` parameter is a string that specifies the name of the file to save the
        plot as. If `name` is not provided, the plot will not be saved
        """
        n_samples = len(self.sample)
        fig = plt.figure(layout='constrained', figsize=(10, 4))
        subfigs = fig.subfigures(2, n_samples, wspace=0.07)
        # Reconstruct sample
        rec_sample = self.sample.to(self.device)
        rec_sample = self.encoder(rec_sample)
        rec_sample = self.decoder(rec_sample)

        for i in range(n_samples):
            # Plot original image
            ax = subfigs[0,i].subplots(1, 1)
            ax.imshow(self.sample[i].permute(1, 2, 0),cmap='gray')
            # Hide X and Y axes label marks
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Original image {0}'.format(i))

            # Recosntruct and plot image
            ax = subfigs[1,i].subplots(1, 1)
            ax.imshow(rec_sample[i].cpu().detach().permute(1, 2, 0).numpy(),cmap='gray')
            # Hide X and Y axes label marks
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Reconstructed image {0}'.format(i))

        if name is not None:
            plt.savefig(
                os.path.join(save_path,name),
                dpi=400
            )
        # return fig

    def encode_data_set(self, name='embeddings.pickle',save_path=None, data_path=None):
        """
        The `encode_data_set` function takes a directory of images, encodes them using a pre-trained
        model, and saves the embeddings along with the image names in a pickle file.
        
        :param name: The name of the output file that will be saved. By default, it is set to
        'embeddings.pickle', defaults to embeddings.pickle (optional)
        :param save_path: The `save_path` parameter is the directory where you want to save the encoded
        data set file. If you don't provide a value for `save_path`, the file will be saved in the
        current working directory
        """
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.img_shape),
                transforms.ToTensor(),
            ]
        )
        if data_path is None: data_path = os.path.join(self.data_dir,'0')
        names = os.listdir(data_path)
        embeddings = torch.zeros(1)
        print("Attempting data embedding...")
        for i in tqdm(range(len(names))):
            img = Image.open(os.path.join(data_path,names[i]))
            img = torch.unsqueeze(transform(img),0)
            enc = self.encoder(img)
            if len(embeddings.shape) <= len(enc.shape):
                N = torch.Size([len(names)])
                embeddings = torch.zeros(N+enc.shape)
                embeddings.shape
            embeddings[i] += enc.detach().numpy()
        
        
        
        embeddings = {
            "names":names,
            "embeddings":embeddings
        }

        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("""succesfully ecnoded images!
            Output file saved at: {0}""".format(
                save_file
            ))
        

# The `Clustering` class performs clustering on input data using the HDBSCAN algorithm and provides
# methods to save the clustered embeddings and move images to clusters based on their labels.
class Represent():
    def __init__(self,input_data):
        """
        The above function is the constructor method for a class that inherits from another class.
        """
        super().__init__()
        self.input_data = input_data['represent']
        self.projector = None

    def project_data(self,data,save_path=None,name="clustered_embeddings.pickle",return_projector = False):
        """
        This function performs dimensionality reduction on input data embeddings using a specified
        algorithm and saves the results to a pickle file.
        
        :param data: The `data_project` function takes in three parameters: `data`, `save_path`, and
        `name`. The `data` parameter is a dictionary containing the embeddings that will undergo
        dimensionality reduction. The function then performs dimensionality reduction on the embeddings
        using the specified algorithm and parameters, saves the resulting
        :param save_path: The `save_path` parameter in the `data_project` function is used to specify
        the directory path where the output file will be saved. If `save_path` is not provided, the
        default value is `None`, which means that the file will be saved in the current working
        directory
        :param name: The `name` parameter in the `data_project` function is a string that represents the
        name of the output file where the processed data will be saved. By default, the value of `name`
        is set to "clustered_embeddings.pickle". However, you can provide a different name when calling
        the, defaults to clustered_embeddings.pickle (optional)
        :return: the `data` dictionary after performing dimensionality reduction on the embeddings and
        saving the results to a pickle file at the specified `save_path` with the default name
        "clustered_embeddings.pickle".
        """
        print("Starting dimensionality reduction via {0}".format(self.input_data['project_algorithm']))
        print("Dimensionality reduction parameters: \n",self.input_data['project_params'])
        features = data['embeddings']
        if len(features.shape) >= 2:
            features = features.reshape(-1,features.shape[-1])

        if self.projector:
            try:
                projection = self.projector.transform(features)
            except:
                projection = self.projector.fit_transform(features)
        else:
            self.projector = dr_algorithms[self.input_data['project_algorithm']](**self.input_data['project_params'])
            projection = self.projector.fit_transform(features)
        data['projection'] = projection

        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("""Output file saved at: {0}""".format(
                save_file
            )
        )
        print('Dimensionality reduction done!')
        return data

    def represent_states(self,data,save_path=None,name="clustered_embeddings.pickle"):
        """
        This Python function performs clustering on input data embeddings using a specified algorithm
        and saves the results to a pickle file.
        
        :param data: The `represent_states` function takes in the following parameters:
        :param save_path: The `save_path` parameter in the `represent_states` function is used to
        specify the directory path where the output file will be saved. If `save_path` is not provided,
        the default value is `None`, which means the file will be saved in the current working directory
        :param name: The `name` parameter in the `represent_states` function is a string that specifies
        the name of the output file where the clustered embeddings will be saved. The default value for
        `name` is "clustered_embeddings.pickle". If you do not provide a specific name when calling the
        function, it will, defaults to clustered_embeddings.pickle (optional)
        :return: the updated data dictionary after clustering, which now includes the 'labels' key with
        cluster assignments.
        """
        print("""Starting clustering via {0}.""".format(self.input_data['cluster_algorithm']))
        print("Clustering parameters: \n",self.input_data['cluster_params'])
        
        features = data['embeddings']
        if len(features.shape) >= 2:
            features = features.reshape(-1,features.shape[-1])
        
        if 'metric' in self.input_data['cluster_params'].keys():
            if self.input_data['cluster_params']['metric'] == 'cosine':
                features = pairwise_distances(features,metric='cosine')
                self.input_data['cluster_params']['metric'] = 'precomputed' 
        representer = cluster_algorithms[self.input_data['cluster_algorithm']](**self.input_data['cluster_params'])
        representer.fit(features)
        print("""Done!""")
        data['labels'] = representer.labels_

        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("""Ourput file saved at: {0}""".format(
                save_file
            ))
        print("""{0} clusters identified (excluding noise)""".format(
            len(np.unique(data['labels']))-1
        ))
        print('Clustering done!')
        return data

    def get_centroids(self,data,metric="cosine",save_path=None,name="clustered_embeddings.pickle"):
        if len(data['embeddings'].shape) >= 2:
            data['embeddings'] = data['embeddings'].reshape(-1,data['embeddings'].shape[-1])
        
        labels = np.unique(data['labels'])
        n_centroids = len(labels)
        if -1 in labels:
            n_centroids -= 1
        
        centroids = np.zeros(shape=n_centroids,dtype=int)
        for l in labels:
            if l != -1:
                idx = np.where(data['labels'] == l)[0]
                pair_dist = pairwise_distances(data['embeddings'][idx],metric=metric)
                centroids[l] += idx[np.argmin(pair_dist.mean(1))]
        data['centroids'] = centroids
        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return  data

    def move_image_to_clusters(self,data,source=None,destination=None):
        """
        The function moves images to clusters based on their labels.
        
        :param data: The `data` parameter is a dictionary that contains information about the images. It
        has the following keys:
        :param source: The `source` parameter is the directory path where the images are currently
        located. By default, it is set to the 'data/ae/0' directory in the current working directory
        :param destination: The `destination` parameter is the path where the images will be moved to.
        If no destination is provided, it will default to a folder named "clusters" inside the current
        working directory
        """
        print("""Moving images to clusters.""")
        if destination is None:
            destination = os.path.join(
                os.getcwd(),
                'data',
                'clusters'
            )
        if source is None:
            source = os.path.join(
                os.getcwd(),
                'data',
                'ae',
                '0'
            )
        for i in np.unique(data['labels']):
            dst = os.path.join(destination,str(i))
            print("Moving files to: ",dst)
            if not os.path.isdir(dst):
                os.makedirs(dst)
            idx = np.where(data['labels'] == i)
            for j in tqdm(idx[0]):
                src = os.path.join(
                    source,
                    data['names'][j]
                )
                shutil.copy(src,dst)

# This class `SiNN` is a neural network module in PyTorch that dynamically constructs a model based on
# input data configuration and performs forward pass through the model.
class  SiNN(nn.Module):
    def __init__(self, input_data):
        """
        This Python function initializes a neural network model based on input data provided.
        
        :param input_data: It looks like the code snippet you provided is an `__init__` method of a
        class that initializes a neural network model based on the input data provided. The input data
        seems to contain information about the layers to be added to the model
        """
        super().__init__()

        self.input_data = input_data

        self.model = nn.Sequential()
        self.in_feat = 0
        for key in self.input_data['sinn'].keys():
            self.model.append(
                layers[self.input_data['sinn'][key]["type"]](
                    **self.input_data['sinn'][key]['params']
                )
            )
            if self.input_data['sinn'][key]["type"] == "Linear":
                self.in_feat =  self.input_data['sinn'][key]["params"]["out_features"]

        if self.in_feat == 0:
            self.in_feat = self.input_data['flatten_size']
            self.model.append(layers["Flatten"](**{"start_dim":1}))
        self.model.append(layers["Linear"](**{"in_features":self.in_feat, "out_features":self.input_data['embedding_size']}))
        if bool(self.input_data["verbose"]) == True:
            print(self.model)

    def forward(self, x):
        """
        The `forward` function takes an input `x` and passes it through the model to generate an output.
        
        :param x: The parameter `x` in the `forward` method is typically the input data that is passed
        to the model for prediction or inference. It could be a single data point or a batch of data
        points, depending on the implementation
        :return: The `forward` method is returning the output of the `model` when it is given input `x`.
        """
        
        return self.model(x)
    
    def triple_forward(self,x1,x2,x3):
        return self.forward(x1), self.forward(x2), self.forward(x3)

# The `Refine` class in Python is designed to refine and train a Siamese Neural Network (SiNN) model
# for triplet mining and clustering tasks, with methods for loading data, training, encoding data,
# getting centroids, cleaning clusters, and merging clusters.
class Refine():
    def __init__(self,input_data):
        """
        The function initializes various parameters and creates a model based on the input data
        provided.
        
        :param input_data: The `input_data` parameter seems to be a dictionary containing various
        configuration settings for your model. Here is a breakdown of the key components extracted from
        `input_data` in your `__init__` method:
        """
        super().__init__()
        self.input_data = input_data
        self.embedding_size = input_data['refine']['embedding_size']
        self.channels = input_data['refine']['channels']
        self.image_size = input_data['refine']['image_size']
        self.bi_thr =  input_data['refine']['binary_threshold']

        # Check if the GPU is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        # Create the model
        self.sinn = SiNN(self.input_data['refine'])

        self.train_params = self.input_data['refine']['train_params']
        self.margin = self.train_params["loss_fn"]['loss_params']['margin']
        self.metric = self.train_params["loss_fn"]['metric']

        self.data_dir = self.input_data['refine']['data_path']

    def load_data(self,data,sample_size=5):
        """
        The function `load_data` processes image data to generate triplets for training a model on image
        embeddings.
        
        :param data: The `data` parameter in the `load_data` method seems to be a dictionary-like object
        containing the following keys:
        :param sample_size: The `sample_size` parameter in the `load_data` method determines how many
        positive and negative samples are selected for each anchor image within a cluster. It specifies
        the number of similar images (positives) and dissimilar images (negatives) to be considered when
        creating triplets for training a model, defaults to 5 (optional)
        """
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.image_size),
                # v2.RandomHorizontalFlip(p=0.85),
                # v2.RandomVerticalFlip(p=0.85),
                # v2.RandomRotation(degrees=270),
                transforms.ToTensor(),
            ]
        )
        labels = np.unique(data['labels'])
        n_centroids  = len(labels)
        if -1 in labels:
            n_centroids -= 1
        
        ordered_idx = []
        self.triplet_idx = []
        # iterate over all the clusters
        for l in range(n_centroids):
            idx = np.where(data['labels'] == l)[0] # retrieves images inside a cluster
            distances = pairwise_distances(X=data['embeddings'][idx],Y=data['embeddings'][data['centroids'][l]].reshape(1, -1),metric=self.metric) # gets distances from centroid
            ordered_idx.append(idx[np.argsort(distances.flatten())].tolist())
        
        
        # there may be a way to improve the performance of this (or not) this will generate 
        # n_centroids*sample_size*(n_centroids-1)*sample_size triplets... I think...
        for a in range(n_centroids):
            anchor = ordered_idx[a][0]
            anchor_name = data['names'][anchor]
            for p in range(sample_size):
                positive = ordered_idx[a][p+1]
                for c in range(n_centroids):
                    if c != a:
                        for n in range(sample_size):
                            negative = ordered_idx[c][n]
                            self.triplet_idx.append([anchor,positive,negative])
        self.original_triplet_idx = np.array(self.triplet_idx) # contains the idx of the triplets in the original dict
            
        # Finally we translate the idx to the ones of the set of unique images
        self.triplet_idx = np.zeros(self.original_triplet_idx.shape,dtype=np.int64)
        unique_idx = list(set(self.original_triplet_idx.flatten()))
        idx_translator = {unique_idx[i]:i for i in range(len(unique_idx))}
        for i in range(len(self.triplet_idx)):
            self.triplet_idx[i,0] = idx_translator[self.original_triplet_idx[i,0]]
            self.triplet_idx[i,1] = idx_translator[self.original_triplet_idx[i,1]]
            self.triplet_idx[i,2] = idx_translator[self.original_triplet_idx[i,2]]
        
        # now we store the data into a dict so we can retrieve the 
        # unique images for the forward pass
        self.names = np.array(data['names'])[unique_idx]
        self.imgs = torch.zeros(len(self.names), self.channels, self.image_size[0], self.image_size[1])
        for i in range(len(self.names)):
            self.imgs[i] += torch.unsqueeze(transform(Image.open(os.path.join(self.data_dir,self.names[i]))),0)[0]
        
    def get_batches(self,batch_size=32):
        """
        The function `get_batches` performs centroid-based triplet mining on embeddings to classify
        triplets based on distance metrics and margin.
        
        :param batch_size: The `batch_size` parameter in the `get_batches` method specifies the number
        of samples to include in each batch during centroid-based triplet mining. It determines how many
        anchor, positive, and negative samples will be processed together in each iteration of the
        mining process, defaults to 32 (optional)
        """
        print("Performing centroid based triplet mining...")
        # Get embeddings
        embeddings = self.sinn.forward(self.imgs)

        # Classify the triplets
        anchor = embeddings[self.triplet_idx[:,0]]
        positive = embeddings[self.triplet_idx[:,1]]
        negative = embeddings[self.triplet_idx[:,2]]        
        
        self.triplet_labels = torch.zeros((self.triplet_idx.shape[0]))
        pos_dist = distances[self.metric](anchor,positive)
        neg_dist = distances[self.metric](anchor,negative)
        semi_hard_1 = pos_dist < neg_dist 
        semi_hard_2 = neg_dist < pos_dist + self.margin
        semi_hard = torch.mul(semi_hard_1,semi_hard_2)
        hard = neg_dist < pos_dist
        self.triplet_labels += semi_hard.long() + 2*hard.long()

        # Retrieve semi-hard and hard triplets
        batch_idx = np.where(self.triplet_labels != 0)[0]
        print("Found {} non easy triplets!".format(len(batch_idx)))
        if len(batch_idx) == 0:
            batch_idx = np.random.randint(low=0,high=self.triplet_idx.shape[0]-1,size=batch_size)
        elif len(batch_idx) < batch_size:
            extra_samples = batch_size - len(batch_idx)
            random_idx = np.random.randint(low=0,high=self.triplet_idx.shape[0]-1,size=extra_samples)
            batch_idx = np.append(batch_idx,random_idx)
        else: pass
        np.random.shuffle(batch_idx)

        num_samples = batch_idx.shape[0]
        num_batches = num_samples // batch_size
        batches = []
        for i in range(num_batches):
            batches.append(batch_idx[i*batch_size:(i+1)*batch_size])

        return batches

    def train(
        self,batch_size=32,num_epochs = 30, 
        seed = 0,early_stops=[],save_path='./',
        name='checkpoint', chkpt_type='best'
        ):
        ### Define the loss function
        self.loss_fn = losses[self.train_params['loss_fn']['loss_type']](**self.train_params['loss_fn']['loss_params'])
        torch.manual_seed(seed)

        params_to_optmize = [
            {'params': self.sinn.parameters()}
        ]

        self.optimizer = optims[self.train_params['optimizer']['type']](
            params=params_to_optmize,
            **self.train_params['optimizer']['params']
        )
        
        best_loss = math.inf

        # Move both the encoder and the decoder to the selected device
        self.sinn.to(self.device)
        diz_loss = {'train_loss':[],'easy':[],'semi-hard':[],'hard':[]}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(batch_size=batch_size)
            for es in early_stops:
                if es.early_stop(train_loss):
                    break
            easy_triplets = len(np.where(self.triplet_labels == 0)[0])
            semihard_triplets = len(np.where(self.triplet_labels == 1)[0])
            hard_triplets = len(np.where(self.triplet_labels == 2)[0])
            print('\n EPOCH {}/{} \t train loss {} \t easy triplets {} \t semi-hard triplets {} \t hard triplets {}'.format(
                epoch + 1, num_epochs,train_loss,easy_triplets,semihard_triplets,hard_triplets))
            diz_loss['train_loss'].append(train_loss)
            diz_loss['easy'].append(easy_triplets)
            diz_loss['semi-hard'].append(semihard_triplets)
            diz_loss['hard'].append(hard_triplets)

            # Save dictionary
            save_file = os.path.join(save_path, "sinn_learning.pickle")
            with open(save_file, 'wb') as handle:
                pickle.dump(diz_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # save checkpoints
            if chkpt_type == "epoch":
                save_name = "sinn_{0}_{1}_epoch.pt".format(name,epoch)
                save_model(model=self.sinn,PATH=save_path,name=save_name)
                
            elif chkpt_type == "best" and best_loss > train_loss:
                save_name = "sinn_{0}_{1}_epoch.pt".format(name,chkpt_type)
                save_model(model=self.sinn,PATH=save_path,name=save_name)
                
                best_loss = train_loss
            else: pass
            
            # Plot losses
            plot_loss(loss={'train_loss':diz_loss['train_loss']},path=save_path, name='sinn_learning.png')
            
            pass_dict = {'easy':diz_loss['easy'],'semi-hard':diz_loss['semi-hard'],'hard':diz_loss['hard']}
            plot_loss(loss=pass_dict,path=save_path, name='sinn_learning_triplets.png')

    def train_epoch(self,batch_size):
        """
        This function trains a neural network model using triplet loss for one epoch.
        
        :param batch_size: The `batch_size` parameter in the `train_epoch` function represents the
        number of samples to process in each iteration during training. It is used to divide the dataset
        into batches to improve efficiency and speed during the training process
        :return: The `train_epoch` function returns the mean of the training loss calculated during the
        epoch.
        """
        self.sinn.train()
        train_loss = []
        for batch_idx in tqdm(self.get_batches(batch_size)):
            # print(self.triplet_idx[np.array(batch_idx)])
            batch = self.imgs[self.triplet_idx[np.array(batch_idx)]].to(self.device) # also making train reconstruct triplets based on their values and indecies of batch indices
            self.optimizer.zero_grad()
            anchor = batch[:,0,:,:,:]
            positive = batch[:,1,:,:,:]
            negative = batch[:,2,:,:,:]
            anchor, positive, negative = self.sinn.triple_forward(anchor, positive, negative)
            loss = self.loss_fn(anchor,positive,negative)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)
    
    def encode_data_set(self, data, name='embeddings.pickle',save_path=None,data_dir=None):
        """
        This function encodes image data using a neural network model and saves the embeddings to a
        pickle file.
        
        :param data: The `data` parameter in the `encode_data_set` method is a dictionary containing
        information about the dataset. It should have a key 'names' which is a list of file names of
        images in the dataset that need to be encoded
        :param name: The `name` parameter in the `encode_data_set` function is a string that specifies
        the name of the file where the encoded data will be saved. By default, it is set to
        'embeddings.pickle', defaults to embeddings.pickle (optional)
        :param save_path: The `save_path` parameter in the `encode_data_set` function is used to specify
        the directory where the encoded data file will be saved. It represents the path where the output
        file will be stored after encoding the data
        :param data_dir: The `data_dir` parameter in the `encode_data_set` function is used to specify
        the directory where the image data is located. If `data_dir` is not provided, the function will
        default to using the `self.data_dir` attribute. This parameter allows flexibility in specifying
        the location of the
        :return: The `encode_data_set` method returns the `data` dictionary after adding the embeddings
        and saving the file.
        """
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )
        if data_dir == None:
            path = self.data_dir
        else:
            path = data_dir
        names = data['names']
        embeddings = torch.zeros(len(names), self.embedding_size)
        print("Attempting data embedding...")
        for i in tqdm(range(len(names))):
            img = torch.unsqueeze(transform(Image.open(os.path.join(path,names[i]))),0)[0]
            img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
            embeddings[i] += self.sinn.forward(img).detach().numpy()
        data['embeddings'] = embeddings
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
    
    def get_centroids(self,data,metric="cosine",save_path=None,name="clustered_embeddings.pickle"):
        """
        The function `get_centroids` calculates centroids for clustered embeddings and saves the result
        in a pickle file.
        
        :param data: The `data` parameter in the `get_centroids` function seems to be a dictionary-like
        object with keys 'labels' and 'embeddings'. The 'labels' key likely contains the cluster labels
        for each data point, and the 'embeddings' key likely contains the embeddings of the data points
        :param metric: The `metric` parameter in the `get_centroids` function specifies the distance
        metric to be used when calculating pairwise distances between embeddings. In this case, the
        default metric is set to "cosine", which means that cosine similarity will be used to calculate
        distances between embeddings, defaults to cosine (optional)
        :param save_path: The `save_path` parameter in the `get_centroids` function is the directory
        path where you want to save the clustered embeddings pickle file. It should be a string
        representing the directory path where you want to save the file. For example, it could be
        something like "/path/to/save/directory
        :param name: The `name` parameter in the `get_centroids` function is used to specify the name of
        the file where the clustered embeddings will be saved. By default, the name is set to
        "clustered_embeddings.pickle" if no specific name is provided when calling the function,
        defaults to clustered_embeddings.pickle (optional)
        :return: the updated `data` dictionary after calculating centroids for each cluster and saving
        the result to a pickle file specified by `save_path` and `name`.
        """
        if metric is None:
            metric=self.metric
        
        labels = np.unique(data['labels'])
        n_centroids = len(labels)
        if -1 in labels:
            n_centroids -= 1
        
        centroids = np.zeros(shape=n_centroids,dtype=int)
        for l in labels:
            if l != -1:
                idx, = np.where(data['labels'] == int(l))
                pair_dist = pairwise_distances(data['embeddings'][idx],metric=metric)
                centroids[int(l)] += idx[np.argmin(pair_dist.mean(1))]
        data['centroids'] = centroids
        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return  data
    
    def clean_cluster(
        self, data:dict, centroids = None, centroid_amount:int = 1, save_path=None, 
        name="sinn_cleaned_embeddings.pickle",clean_noise=True
        ) -> np.ndarray: 
        """
        The `clean_cluster` function takes in a dictionary of data, removes outliers, relabels clusters
        based on distances, and saves the cleaned data to a pickle file.
        
        :param data: The `data` parameter is a dictionary containing the following keys:
        :type data: dict
        :param centroid_amount: The `centroid_amount` parameter in the `clean_cluster` method specifies
        the number of centroids to be used for clustering the data. It is used to divide the distances
        matrix into clusters based on the specified number of centroids, defaults to 1
        :type centroid_amount: int (optional)
        :param save_path: The `save_path` parameter in the `clean_cluster` function is used to specify
        the directory path where the cleaned data will be saved. It is the location where the pickle
        file containing the cleaned data will be stored. You can provide the directory path as a string
        when calling the function to indicate where
        :param name: The `name` parameter in the `clean_cluster` method is a string that specifies the
        name of the file where the cleaned embeddings will be saved. By default, the value of `name` is
        set to "sinn_cleaned_embeddings.pickle". This file will be saved at the location specified by,
        defaults to sinn_cleaned_embeddings.pickle (optional)
        :return: The `clean_cluster` method returns the updated `data` dictionary after cleaning the
        cluster labels and saving the modified data to a pickle file specified by `save_path` and
        `name`.
        """
        
        if centroids == None:
            centroids =  data['embeddings'][data['centroids']]
        # code for relabeling.
        distances : np.ndarray = pairwise_distances(X=data['embeddings'],Y = centroids,metric=self.metric)
        cluster_distances : np.ndarray = np.reshape(distances, (distances.shape[0],distances.shape[1]//centroid_amount, centroid_amount))
        if clean_noise:
            data['labels'] = np.argmin(np.sum(cluster_distances, axis=-1), axis=1)
        else:
            idx = np.where(data['labels'] != -1)[0]
            data['labels'][idx] = np.argmin(np.sum(cluster_distances, axis=-1), axis=1)[idx]

        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return data

    def merge_clusters(self,data:dict,alpha=0.8,n_borders=10,save_path=None,name="sinn_merged_embeddings.pickle"):
        """
        The function `merge_clusters` calculates distances between clusters, merges clusters based on
        certain conditions, and saves the updated data to a pickle file.
        
        :param data: The `data` parameter in the `merge_clusters` function is a dictionary containing
        the following keys:
        :type data: dict
        :param save_path: The `save_path` parameter in the `merge_clusters` function is used to specify
        the directory path where the merged embeddings data will be saved. It is an optional parameter,
        meaning if no path is provided, the default directory will be used for saving the data
        :param name: The `name` parameter in the `merge_clusters` function is a string that specifies
        the name of the file where the merged embeddings will be saved. By default, if no value is
        provided for `name`, it will be set to "sinn_merged_embeddings.pickle". This file will contain
        the updated, defaults to sinn_merged_embeddings.pickle (optional)
        :return: The function `merge_clusters` is returning the updated `data` dictionary after merging
        clusters based on certain criteria.
        """
        
        c_c_dist = pairwise_distances(X=data['embeddings'][data['centroids']])
        closest_cluster = np.argsort(c_c_dist)[:,1]
        d_intra = np.zeros(c_c_dist.shape[0])
        d_inter = np.zeros(c_c_dist.shape[0])
        s_c_dist = pairwise_distances(X=data['embeddings'],Y=data['embeddings'][data['centroids']])
        for l in range(len(data['centroids'])):
            idx = np.where(data['labels'] == l)[0]
            try:
                # d_intra[l] += np.max(s_c_dist[idx,l])
                d_intra[l] +=  np.average(np.sort(s_c_dist[idx,l]).flatten()[-n_borders:])
            except:
                d_intra[l] += 0.0
            d_inter[l] += c_c_dist[l,closest_cluster[l]]
            
        flags = alpha*d_inter > d_intra
        print(closest_cluster)
        merge_dict = {i:closest_cluster[i] for i in range(len(closest_cluster))}
        
        corrected_labels = data['labels'].copy()
        for i in range(len(flags)):
            if flags[i]:
                idx = np.where(data['labels'] == i)[0]
                print("Before",data['labels'][idx])
                data['labels'][idx] = merge_dict[i]*np.ones(len(idx))
                print("After",data['labels'][idx])


        
        save_file = os.path.join(save_path,name)
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return  data
    
    def rename_labels(self,data:dict,save_path=None,name="sinn_new_labels_embeddings.pickle"):
        labels = np.unique(data['labels'])
        n_centroids = len(labels)
        if -1 in labels:
            n_centroids -= 1
            labels = np.delete(np.array(labels),np.where(labels == -1)[0])
        self.label_map = {-1:-1}
        for i in range(n_centroids):
            self.label_map[i] = labels[i]
            idx = np.where(data['labels'] == labels[i])[0]
            data['labels'][idx] = i*np.ones(len(idx))
        
        data = self.get_centroids(data,metric=self.metric,save_path=save_path,name=name)
        
        return  data
