import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import imageio
import io
from io import BytesIO
from utils_jlg.video_image import make_gif_1D_arrays
from config import settings as stts
from tqdm import tqdm

plt.style.use('ggplot')


def prepare_x_y_time_data( simulations , skip_steps = 10, store_steps_ahead = 5):
    
    """
    Prepare states and states ahead datasets
    
    to train a model for inference of a system future states
    
    
    skip steps is the number of frames ahead to skip when  building the datasets
    
    store_steps_ahead is the number of next steps stored in the array Y in order to do multistep prediction
    
    """

    X = []
    Y = []

    for simulation in tqdm(simulations):

        sim = simulation

        lsim = len(sim)

        for i in range(int( np.floor( lsim/skip_steps) )-store_steps_ahead ):

            s = i*skip_steps

            _Y = []


            for j in range(1,store_steps_ahead):
                sj = (i+j)*skip_steps
                _Y.append(sim[sj])

            _Y = np.array(_Y)


            X.append(sim[s])
            Y.append(_Y)

    X = np.array(X)
    Y = np.array(Y)

    return X,Y


def plot_t_x_fun_slices(array, x = np.array([]), title = "", nslices = 10):
    """
    
    Plot slices of a time_steps x Xnodes func, like the solution of an evolutive 1D pde
    
    """
    assert len(np.shape(array))==2,"array must be Tsteps x Xnodes"
    index = np.arange(0,len(array),1)
    _every = int(len(array)/nslices)
    selection = index[::_every]
    
    fig = plt.figure(figsize = (10,10))
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    
    if not(x.any()):
        x = np.linspace(0,1,np.shape(array)[1])
    for i in selection:

        ax.plot(array[i])
        
    return fig


def post_process_save_pde_1d(array, name, save_dir):
    
    
    save_name_numpy = os.path.join(save_dir,name)
    np.save(save_name_numpy, array)
    
    save_name_figure = os.path.join(save_dir,name+".png")
    fig1 = plot_t_x_fun_slices(array)
    fig1.savefig(save_name_figure)
    
    #save_name_gif = os.path.join(save_dir, name+".gif")
    #make_gif_1D_arrays(array[::20], duration = 0.1, name = save_name_gif)
    
    
def load_arrays(name):
    
    dir_data = stts["dir_data"]
    dir_name = os.path.join(dir_data,name)
    
    _dirs = os.listdir(dir_data)
    
    assert name in _dirs, "Only found {} dirs, not {}".format(_dirs, name)
    
    _files = os.listdir(dir_name)
    _arrays = [_file for _file in _files if ".npy" in _file]
    
    list_arrays = [np.load( os.path.join(dir_name,array) ) for array in tqdm(_arrays)]
    
    return list_arrays