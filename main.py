from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from models import DataModulePL, ModelEvaluationCallback, BaseModelPL, FNO_1d_time_pl, BasicNet_pl
from fire import Fire
from config import settings as stts
import os
import pytorch_lightning as pl
import yaml
from utils import load_model,get_data_x0_inverse_problem
from inverse import FindInitialCondition

config_dict = {

    "data": dict(batch_size = 120, skip_steps = 40, store_steps_ahead = 5,
                 test_ratio = 0.25, max_time_index = 600, drop_last = True, skip_first_n = 8 ),

    "width":30,
    "modes":20,
    "epochs":400,
    "name_experiment": "default_experiment_name",
    "name_data": "AC1d",
    "period_evaluation_epochs":50,
    "tol_next_step": 1e-4,
    "weight_decay": 1e-6,
    "load_pre_trained_model": False,
    "model":"fno",
    "norm":True,

}

cfg = config_dict


config_dict_inverse = {
    "simulation_n": 33,
    "step_x0": 70,
    "steps_ahead": 80,
    "skip_steps": 40,
    "model_name": "AC1d_fno_small",
    "reg_grad_x": 1e-2,
    "name_experiment": "default_inverse_experiment_name",
    "max_iter": 2000,
    "tol":5e-3,
}

cfg_inv = config_dict_inverse

def save_experiment_params(experiment_cfg, save_dir):

    filename = os.path.join(save_dir, "experiment_params.yaml")
    with open(filename,'w') as f:
        yaml.dump(experiment_cfg, f)

class Main():

    def run_experiments_1D_time(self):

        width_list = [10,100]
        modes_list = [10,40]

        for width in width_list:
            for modes in modes_list:
                name_experiment = "modes_{}_width_{}".format(modes,width)
                self.train_1D_time(modes = modes, width = width, name_experiment = name_experiment)

    def train_1D_time(self, **kwargs):



        for key,value in kwargs.items():
            if key in cfg.keys():
                cfg[key] = value
            elif key in cfg["data"].keys():
                cfg["data"][key] = value
            else:
                raise(ValueError("only admited params are {}".format(config_dict)))


        print("Training with parameters {}".format(cfg))
        data = DataModulePL(cfg["name_data"], **cfg["data"])
        logs_dir = stts["dir_logs"]
        models_checkpoints_dir = os.path.join(logs_dir, "model_checkpoint")
        results_dir = os.path.join(stts["dir_results"], cfg["name_data"]+"_"+cfg["name_experiment"])
        callback1 = ModelEvaluationCallback(data, results_dir, period_evaluation_epochs=cfg["period_evaluation_epochs"])
        callback2 = ModelCheckpoint(dirpath= os.path.join(results_dir, "checkpoints"),
                    filename="{epoch:02d}",
                     verbose=True, save_last=True)

        try:
            os.makedirs(results_dir)
        except:
            pass


        logger_csv = CSVLogger(logs_dir, name= cfg["name_data"]+"_"+cfg["name_experiment"])
        logger_tensorboard = TensorBoardLogger(logs_dir, name= cfg["name_data"]+"_"+cfg["name_experiment"])


        #callback = SimEvalCallback(datamod, results_dir,save_every = 10)
        #early_stopping = EarlyStopping('val_loss', patience = 10, min_delta = 1e-4)

        trainer = pl.Trainer(max_epochs = cfg["epochs"], callbacks = [callback1, callback2], flush_logs_every_n_steps = 20, log_every_n_steps= 20,
                            logger = [logger_csv,logger_tensorboard],default_root_dir = models_checkpoints_dir, gpus = 1)

        if cfg["load_pre_trained_model"]:
            print("\n### LOADING PRETRAINED MODEL {}".format(os.path.join(results_dir, "checkpoints/last.ckpt")))

            if cfg["model"] == "fno":
                model = load_model(FNO_1d_time_pl,os.path.join(results_dir, "checkpoints/last.ckpt"))
            elif cfg["model"] == "bnet":
                model = load_model(BasicNet_pl,os.path.join(results_dir, "checkpoints/last.ckpt"))
        else:
            if cfg["model"] == "fno":
                model = FNO_1d_time_pl(cfg["modes"], cfg["width"],norm = cfg["norm"], results_dir = results_dir, tol_next_step = cfg["tol_next_step"], weight_decay = cfg["weight_decay"])
            elif cfg["model"] == "bnet":
                print("Using bnet")
                model = BasicNet_pl(1,cfg["width"],1,results_dir = results_dir, tol_next_step = cfg["tol_next_step"], weight_decay = cfg["weight_decay"] )

        save_experiment_params(cfg, results_dir)

        trainer.fit(model, datamodule = data)
        
    def find_initial_condition(self,**kwargs):
        
        for key,value in kwargs.items():
            if key in cfg_inv.keys():
                cfg_inv[key] = value

            else:
                raise(ValueError("only admited params are {}".format(cfg_inv)))
        
  
        name_data = "AC1d"
        simulation = cfg_inv["simulation_n"]
        step_x0 = cfg_inv["step_x0"]
        steps_ahead = cfg_inv["steps_ahead"]
        model_steps_ahead = int(steps_ahead/cfg_inv["skip_steps"])
        model = load_model(FNO_1d_time_pl,"./results/{}/checkpoints/last.ckpt".format(cfg_inv["model_name"]))
        reg_grad_x = cfg_inv["reg_grad_x"]
        name = cfg_inv["name_experiment"]+"_simulation_{}_step_{}_ahead_{}".format(simulation,step_x0, steps_ahead)
        
        
        x0_ground_truth, xnext = get_data_x0_inverse_problem(name_data, simulation, step_x0,steps_ahead)



        finder = FindInitialCondition(model, name =name)

        out = finder.run(xnext,2,x0_ground_truth=x0_ground_truth, log_every = 20, tol = cfg_inv["tol"], reg_grad_x = reg_grad_x, max_iter = cfg_inv["max_iter"])


if __name__ == "__main__":
    Fire(Main)
