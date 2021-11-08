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
import torch
from itertools import cycle
import multiprocessing

plt.style.use('ggplot')

class Dummylogging:
    debug = print
    info = print
    warning = print

logging = Dummylogging()

def get_data_x0_inverse_problem(name_data, simulation, step_x0,steps_ahead):

    arrays = load_arrays(name_data)
    x0 = arrays[simulation][step_x0][None,None,:]
    xnext = arrays[simulation][step_x0+steps_ahead][None,None,:]


    return x0, xnext


def load_model(model_class, checkpoint_pl_filename):
    ###doing this because load_from_checkpoint in pytorch lightining doesnt load model weights ( maybe has to do with inheritance )

    ckpt = torch.load(checkpoint_pl_filename)

    hparams = list(ckpt["hyper_parameters"].values())
    state_dict = ckpt["state_dict"]
    model = model_class(*hparams)

    model.load_state_dict(state_dict)

    logging.info("Loading model {} with hparams {} epoch {}".format(model_class, ckpt["hyper_parameters"], ckpt["epoch"]))

    return model



def _make_plots_evaluate(pred, real, preffix_name, save_dir,i):

        fig0 = plt.figure(figsize=(10,10))
        plt.plot(real[0], color = "black", label = "u0")

        plt.plot(real[1], color = "green", label = "u1")
        plt.plot(real[2], color = "green", label = "u2")
        plt.plot(real[3], color = "green", label = "u3")

        plt.plot(pred[1], color = "red", label = "pred1")
        plt.plot(pred[2], color = "red", label = "pred2")
        plt.plot(pred[3], color = "red", label = "pred3")
        plt.legend()

        fig0.savefig(os.path.join(save_dir, preffix_name+"_"+"1D_next_step_comparison_sample_{}.png".format(i)))


        fig1 = plot_t_x_fun_slices_comparison([real,pred], names = ["real","pred"], nslices = 2, title = "Ground T. vs Pred")

        fig1.savefig(os.path.join(save_dir, preffix_name+"_"+"1D_plot_comparison_sample_{}.png".format(i)))


        fig2 = plt.figure(figsize = (10,10))

        plt.plot(np.mean(np.abs(pred),axis = 1), color = "red", label = "pred")
        plt.plot(np.mean(np.abs(real),axis = 1), color = "green", label = "real")
        plt.xlabel("Time")
        plt.ylabel("Avg abs phase")
        plt.title("Abs phase, ground truth vs pred")
        plt.legend()

        fig2.savefig(os.path.join(save_dir, preffix_name+"_"+"vs_phase_sample_{}.png".format(i)))

        make_gif_1D_arrays(pred, duration = 0.5, name = os.path.join(save_dir,preffix_name+"_"+"sample_{}.gif".format(i)))

def evaluate_model(model, test_samples, preffix_name = "", save_dir = None):

    """
    Plot evaluation figs for 1d time sim inference
    """

    print("Evaluating model with model.training {}, saving results in {}".format(model.training, model, save_dir))

    try:
        os.makedirs(save_dir)
    except:
        pass

    for i,sample in tqdm(enumerate(test_samples)):

        pred, real = eval_sim(model, sample)

        x = multiprocessing.Process(None, target = _make_plots_evaluate, args = (pred,real,preffix_name,save_dir,i, ))
        x.start()


def eval_sim(model, ground_truth_sim):
    """
    evaluates recurrently model using ground_truth_sim as x0, returns np.array
    """

    model.eval()

    if isinstance(ground_truth_sim,np.ndarray):
        ground_truth_sim = torch.Tensor(ground_truth_sim)
    elif isinstance(ground_truth_sim,torch.Tensor):
        pass
    else:
        raise(ValueError("torch.Tensor or np.ndarray are the only types allowed"))

    assert len(ground_truth_sim.shape) == 3, "ground_truth_sim must be steps x channels x dim"

    x0 = ground_truth_sim[0].unsqueeze(0)
    steps = ground_truth_sim.shape[0]

    #model.eval()

    outs = []

    out = x0

    with torch.no_grad():

        for step in tqdm(range(steps)):

            outs.append(out.cpu().numpy()[0,0])

            out = model(out)

    pred = np.array(outs)
    real = ground_truth_sim.cpu().detach().numpy()[:,0]

    return pred, real




def prepare_x_y_time_data( simulations , skip_steps = 10, store_steps_ahead = 5, max_time_index = 300, skip_first_n = 0):

    """
    Prepare states and states ahead datasets

    to train a model for inference of a system future states


    skip steps is the number of frames ahead to skip when  building the datasets

    store_steps_ahead is the number of next steps stored in the array Y in order to do multistep prediction

    """

    X = []
    Y = []

    simulations = np.array(simulations)

    simulations = simulations[:,skip_first_n:max_time_index,:]

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


    X = X[:,None,:]
    Y = Y[:,:,None,:]

    logging.info("Preparing data with args skip_steps {} max_time {} \n shape X {} shape Y {}".format(skip_steps, max_time_index, np.shape(X), np.shape(Y)))

    return X,Y


def plot_t_x_fun_slices(array, x = np.array([]), title = "", nslices = 10, ax = None,label = "", color = None):
    """

    Plot slices of a time_steps x Xnodes func, like the solution of an evolutive 1D pde

    """
    assert len(np.shape(array))==2,"array must be Tsteps x Xnodes"
    index = np.arange(0,len(array),1)
    _every = int(len(array)/nslices)
    selection = index[::_every]

    if not(ax):
        fig = plt.figure(figsize = (10,10))
        fig.suptitle(title)
        ax = fig.add_subplot(111)

    if not(x.any()):
        x = np.linspace(0,1,np.shape(array)[1])
    for i in selection:
        if i == 0:
            label = label
        else:
            label = None
        ax.plot(x, array[i], color = color, label = label)

    return ax.get_figure()


def plot_t_x_fun_slices_comparison(list_arrays, names = [], x = np.array([]), title = "", nslices = 10):

    """
    y vs t plot comparison of the arrrays in list_arrays, must have the same length

    """

    fig = plt.figure(figsize = (10,10))
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    if names:
        assert len(names)==len(list_arrays), "names and list arrays havent the same length"
    else:
        names = np.arange(0,len(list_arrays),1)

    colors = cycle(["green","red","blue","yellow","black","brown","pink"])

    for i,array in enumerate(list_arrays):

        color = next(colors)
        fig = plot_t_x_fun_slices(array, x = x, title = title, nslices = nslices, ax = ax, label = "{}".format(names[i]), color = color)
    ax.legend()


    return fig


def post_process_save_pde_1d(array, name, save_dir):


    save_name_numpy = os.path.join(save_dir,name)
    np.save(save_name_numpy, array)

    save_name_figure = os.path.join(save_dir,name+".png")
    fig1 = plot_t_x_fun_slices(array)
    fig1.savefig(save_name_figure)

    #save_name_gif = os.path.join(save_dir, name+".gif")
    #make_gif_1D_arrays(array[::20], duration = 0.1, name = save_name_gif)


def load_arrays(name, return_names = False):

    dir_data = stts["dir_data"]
    dir_name = os.path.join(dir_data,name)

    _dirs = os.listdir(dir_data)

    assert name in _dirs, "Only found {} dirs, not {}".format(_dirs, name)

    _files = os.listdir(dir_name)
    names = [_file for _file in _files if ".npy" in _file]

    list_arrays = [np.load( os.path.join(dir_name,array) ) for array in tqdm(names)]
    if not(return_names):
        return list_arrays
    else:
        return list_arrays, names
