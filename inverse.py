from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from pyro.infer import MCMC, NUTS
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm import tqdm
from pyro.optim import Adam
from pyro.infer import Predictive
import pyro.distributions as dist
from pyro.nn.module import PyroModule
import pyro

class FindInitialCondition():

    """
    Find initial condition by gradient descent on network input
    """

    def __init__(self, model_forward, loss = torch.nn.L1Loss, optimizer = torch.optim.Adam, name = "inverse_initial_condition",log_dir = None):

        assert hasattr(model_forward, "recurrent_evaluation"), "Model forward must have the method recurrent evaluation with args(x0,steps) that returns list with outputs at each timestep"

        self._model = model_forward
        self._criterion = loss()
        self._optimizer = optimizer
        self._name = name
        if not(log_dir):
            log_dir = os.path.join(stts["dir_logs"], name)
        self.writer = SummaryWriter(log_dir,flush_secs = 10)
        self._save_dir = os.path.join(stts["dir_results"], name)

        try:
            os.makedirs(self._save_dir)
        except:
            pass

    def run(self, xtarget, step_ahead_model,x0_ground_truth = None,reg_grad_x = 1e-3,  x0_initial = None, lr = 1e-2, max_iter = 2000, tol = 5e-3, verbose = True, log_every = 100):

        if not(isinstance(x0_initial,torch.Tensor) or isinstance(x0_initial,np.ndarray)):
            x0_initial = np.random.rand(*xtarget.shape)-0.5

        if not(isinstance(x0_ground_truth,torch.Tensor) or isinstance(x0_ground_truth,np.ndarray)):
            x0_ground_truth = None
        else:
            x0_ground_truth = torch.Tensor(x0_ground_truth)

        x0 = torch.nn.Parameter(torch.Tensor(x0_initial) )
        xtarget = torch.Tensor(xtarget)
        self._xtarget = xtarget

        optimizer = self._optimizer((x0,), lr = lr)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.8, patience = 100, threshold = 1e-4,threshold_mode = "abs", min_lr = 1e-4, verbose = True)

        self._model.eval()

        for i in tqdm(range(max_iter)):

            out_dict = self.step(scheduler, step_ahead_model, xtarget, x0,reg_grad_x)

            if out_dict["loss"]<tol:
                out_dict["success"] = True
                break


            if i%log_every==0:
                self._log_results(i,x0_initial,   optimizer.param_groups[0]["lr"],reg_grad_x,step_ahead_model, out_dict, self.writer, x0_ground_truth = x0_ground_truth, verbose = verbose)

        if (i+1) >= max_iter:
            out_dict["success"] = False
        self._log_results(i,x0_initial,  optimizer.param_groups[0]["lr"],reg_grad_x,step_ahead_model, out_dict, self.writer, x0_ground_truth = x0_ground_truth, verbose = verbose, final = True)


        return out_dict


    def _log_results(self,step,x0_initial, lr, reg_grad_x, step_ahead_model,  out_dict, writer, x0_ground_truth = None, verbose = True, final = False):

        writer.add_scalar("Loss/initial_condition", out_dict["loss"], global_step = step)
        writer.add_scalar("Loss_reconstruction/initial_condition", out_dict["loss_reconstruction"], global_step = step)
        writer.add_scalar("Loss_reg_grad/initial_condition", out_dict["loss_reg_grad"], global_step = step)
        writer.add_scalar("lr",lr, step)
        writer.add_scalar("reg_grad_x", reg_grad_x, step)
        fig = plt.figure()

        if isinstance(x0_ground_truth, torch.Tensor):

            plt.plot(x0_ground_truth[0,0], '--', label = "ground_truth")

        plt.plot(out_dict["x0"].detach().numpy()[0,0], label = "x0 iter {}".format(step))
        plt.legend()

        writer.add_figure("initial_condition_finder", fig, global_step = step)

        if verbose:
            print("step, loss", step, out_dict["loss"])




        if final:

            fig = plt.figure(figsize = (12,12))
            plt.plot(x0_initial[0,0],'--', label = "initial condition optimization", color = "black")
            plt.plot(out_dict["x0"].detach()[0,0], color = "red", label = "found x0")
            if isinstance(x0_ground_truth, torch.Tensor):
                plt.plot(x0_ground_truth[0,0],'--', color = "green", label = "ground_truth")
            plt.legend()
            fig.savefig(os.path.join(self._save_dir,"initial.png"))


            fig = plt.figure(figsize = (12,12))
            if isinstance(x0_ground_truth, torch.Tensor):
                plt.plot(x0_ground_truth[0,0],'--', color = "green", label = "ground_truth")

            xpred = self._model.recurrent_evaluation(out_dict["x0"].detach(), step_ahead_model)[-1].detach()
            plt.plot(out_dict["x0"].detach()[0,0], color = "red", label = "found x0")
            plt.plot(xpred[0,0], color = "red", label = "NN evolution from found x0")
            plt.plot(self._xtarget[0,0],'--', color = "green", label = "ground truth evolution from original x0")
            plt.legend()
            figname = os.path.join(self._save_dir,"x0_results_final.png")

            fig.savefig(figname)
            if isinstance(x0_ground_truth, torch.Tensor):
                fig = plt.figure(figsize = (12,12))
                plt.plot(out_dict["x0"].detach()[0,0], color = "red", label = "found x0")
                plt.plot(x0_ground_truth[0,0],'--', color = "green", label = "ground_truth x0")
                plt.legend()
                fig.savefig(os.path.join(self._save_dir,"x0_gt_x0_found_comparison.png"))


    def step(self, scheduler, step_ahead_model, xtarget, x0,reg_grad_x):

        """
        step_ahead_model is the distance in time steps of the xtarget respect to x0
        """

        optimizer = scheduler.optimizer

        optimizer.zero_grad()

        xpred = self._model.recurrent_evaluation(x0, step_ahead_model)[-1]

        loss_1 = self._criterion(xpred, xtarget)

        filter_array = torch.Tensor([-1,1])[None,None,:]
        grad_x = torch.nn.functional.conv1d(x0,filter_array,padding = 1)[...,:-1]
        loss_2 = torch.sum(torch.square(grad_x))

        loss = loss_1+reg_grad_x*loss_2

        loss.backward()

        optimizer.step()
        scheduler.step(loss)

        return {"x0":x0, "loss": loss, "loss_reconstruction": loss_1, "loss_reg_grad": loss_2}






class BayesianInverse(PyroModule):
    def __init__(self,n_steps, model_forward, sigma_obs = 0.1, sigma_xo = 0.3):
        super().__init__()

        self._model_forward = model_forward
        self._model_forward.eval()
        self._n_steps = n_steps
        self._sigma_obs = sigma_obs
        self._sigma_xo = sigma_xo

    def forward(self, y):

        sigma_obs = self._sigma_obs
        sigma_xo = self._sigma_xo
        xo = pyro.sample("x0",dist.Normal(torch.zeros(y.shape), sigma_xo).to_event(1))
        mean = self._model_forward.recurrent_evaluation(xo, self._n_steps, grad = True)[-1]
        with pyro.plate("data", 1):
            obs = pyro.sample("obs", dist.Normal(mean, sigma_obs).to_event(1), obs= y.reshape(-1,1,1,y.shape[-1]))
        return xo,mean



def find_xo_MCMC(model_forward, name_data, simulation, step_x0, steps_ahead, skip_steps_training,sigma_xo = 0.3,sigma_obs = 0.1, num_samples = 200, warmup_steps = 10, name = "inverse_MCMC"):

    xo_gt, y = get_data_x0_inverse_problem(name_data, simulation, step_x0, steps_ahead)
    xo_gt,y = torch.Tensor(xo_gt),torch.Tensor(y)

    steps_model = int(steps_ahead/skip_steps_training)
    model = BayesianInverse(steps_model, model_forward, sigma_xo = sigma_xo, sigma_obs = sigma_obs)


    nuts_kernel = NUTS(model.forward)
    mcmc = MCMC(nuts_kernel,num_samples,warmup_steps = warmup_steps)
    mcmc.run(y)

    samples = {k: v.detach() for k, v in mcmc.get_samples().items()}

    evaluate_inverse(model_forward, steps_ahead, samples, xo_gt, y,  title_append = "MCMC", name = name)


def find_xo_VI(model_forward, name_data, simulation, step_x0, steps_ahead, skip_steps_training,sigma_xo = 0.3, sigma_obs = 0.1, lr = 0.005, num_iters = 2000, num_samples_posterior = 1500, name = "inverse_VI"):

    xo_gt, y = get_data_x0_inverse_problem(name_data, simulation, step_x0, steps_ahead)
    xo_gt,y = torch.Tensor(xo_gt),torch.Tensor(y)

    steps_model = int(steps_ahead/skip_steps_training)
    model = BayesianInverse(steps_model, model_forward, sigma_xo = sigma_xo, sigma_obs = sigma_obs)
    guide = AutoDiagonalNormal(model.forward)

    svi = SVI(model.forward,
          guide,
          Adam({"lr": lr}),
          loss=Trace_ELBO())

    pyro.clear_param_store()
    num_iters = num_iters
    losses = []
    for i in tqdm(range(num_iters)):
        elbo = svi.step(y)
        if i % 500 == 0:
            print("Elbo loss: {}".format(elbo))
        losses.append(elbo)

    predictive = Predictive(model.forward, guide=guide, num_samples=num_samples_posterior)
    samples = predictive(y)



    evaluate_inverse(model_forward, steps_ahead, samples, xo_gt, y, losses = losses, title_append = "VI", name = name)



def evaluate_inverse(model_forward, steps_ahead, samples, xo_gt, y, losses = None, results_dir = None, name = None,  title_append = ""):

    if not(name):
        name = inverse_problem_steps_ahead
    if not(results_dir):
        results_dir = os.path.join(stts["dir_results"],"{}_{}".format(name,steps_ahead))
    try:
        os.makedirs(results_dir)
    except:
        pass

    fig = plt.figure(figsize=(14,10))
    mean = torch.mean(samples["x0"],axis=0)
    plt.plot(model_forward.recurrent_evaluation(mean,steps_ahead,grad = False)[-1][0,0], color = "black", label = "Network evaluation of avg posterior")
    plt.plot(model_forward.recurrent_evaluation(torch.quantile(samples["x0"],0.75,axis = 0),2,grad = False)[-1][0,0],'--', color = "red", label = "75 25 quantile")
    plt.plot(model_forward.recurrent_evaluation(torch.quantile(samples["x0"],0.25,axis = 0),2,grad = False)[-1][0,0],'--', color = "red")

    plt.plot(y[0,0],"--", color = "green", label = "ground truth evolution")
    plt.title(r"Inverse problem x(T) {}".format(title_append))
    plt.legend()

    fig.savefig(os.path.join(results_dir,"evolution_comparison.png"))

    fig = plt.figure(figsize=(14,10))
    plt.plot(np.mean(samples["x0"].numpy(),axis = 0)[0,0], color = "black", label = "avg posterior samples")
    plt.plot(np.percentile(samples["x0"].numpy(),25,axis = 0)[0,0], '--', color = "red",label = "75 25 quantiles")
    plt.plot(np.percentile(samples["x0"].numpy(),75,axis = 0)[0,0], '--', color = "red")
    plt.plot(xo_gt[0,0],color = "green", label = "ground truth initial condition")
    plt.legend()
    plt.title("Inverse problem x(0) {}".format(title_append))

    fig.savefig(os.path.join(results_dir,"initial_condition_comparison.png"))

    if losses:
        fig = plt.figure(figsize = (10,10))
        plt.plot(losses)
        plt.title("Loss ELBO")
        fig.savefig(os.path.join(results_dir,"loss_elbo.png"))
