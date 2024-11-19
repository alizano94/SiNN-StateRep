import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def plot_loss(loss,name=None,path=None):
    """
    The function `plot_loss` is used to plot the loss values for different keys and save the plot as an
    image if specified.
    
    :param loss: The `loss` parameter is a dictionary that contains the loss values for different keys.
    Each key represents a different loss metric. The values associated with each key are the loss values
    at each epoch
    :param name: The name parameter is used to specify the name of the file to save the plot as. If you
    want to save the plot with a specific name, you can provide it as a string value to the name
    parameter
    :param path: The `path` parameter is the directory path where you want to save the plot image
    """
    fig = plt.figure(layout='constrained', figsize=(10, 4))
    for key in loss.keys():
        plt.plot(
            range(len(loss[key])),
            loss[key],
            label=key
        )
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if name is not None:
        plt.savefig(
            os.path.join(
                path,
                name
            ),
            dpi=400
        )
    plt.close('all')
def plot_2d_clusters(data,name,path):
    pass


def plot_configuration_from_txt(configuration_file,save_file=None,save_path='./',show=False):
    # Read the data from the text file
    with open(configuration_file, 'r') as file:
        lines = file.readlines()

    # Extract the coordinates from each line
    coordinates = []
    for line in lines:
        parts = line.split()
        index = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        coordinates.append([x, y, z])

    # Convert the list of coordinates to a NumPy array
    coordinates_array = np.array(coordinates)
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(
        coordinates_array[:,0],
        coordinates_array[:,1],
        s = 300
    )
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    plt.axis('off')
    
    if save_file is not None:
        save_path = os.path.join(save_path,save_file)
        plt.savefig(save_path,dpi=400)
    
    if show:
        plt.show()
    
    plt.close('all')
        
