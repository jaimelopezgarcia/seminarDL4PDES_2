from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from models import DataModulePL, ModelEvaluationCallback, BaseModelPL, FNO_1d_time_pl
from fire import Fire
from config import settings as stts
import os
import pytorch_lightning as pl


config_dict = {

    "data": dict(batch_size = 120, skip_steps = 20, store_steps_ahead = 5,
                 test_ratio = 0.2, max_time_index = 500, drop_last = True),

    "width":40,
    "modes":12,
    "epochs":200,
    "name_experiment": "default_experiment_name",
    "name_data": "AC1d",
    "period_evaluation_epochs":30,
    "tol_next_step": 0.001,
    "weight_decay": 1e-5

}

cfg = config_dict

class Main():

    def run_experiments_1D_time(self):

        width_list = [10,50,100]
        modes_list = [5,20,50]

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


        model = FNO_1d_time_pl(cfg["modes"], cfg["width"], results_dir = results_dir, tol_next_step = cfg["tol_next_step"], weight_decay = cfg["weight_decay"])

        trainer.fit(model, datamodule = data)


if __name__ == "__main__":
    Fire(Main)
