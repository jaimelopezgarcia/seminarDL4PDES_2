import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
import scipy.io
from utils import load_arrays, prepare_x_y_time_data, evaluate_model
from fno.fourier_1d import FNO1d_time
from config import settings as stts
import yaml

class SimDataset(Dataset):
    def __init__(self, X,Y):
        #self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._X = torch.Tensor(X)

        self._Y = torch.Tensor(Y)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):

        sample = {"X": self._X[idx], "Y": self._Y[idx]}

        return sample

class DataModulePL(pl.LightningDataModule):#problemdatamoduele import

    def __init__(self, name , batch_size = 60, skip_steps = 10,
                 store_steps_ahead = 5, test_ratio = 0.2,
                 max_time_index = 500, drop_last = True, skip_first_n = 0):

        super().__init__()

        self._name = name
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size

        self._skip_steps = skip_steps
        self._store_steps_ahead = store_steps_ahead
        self._test_ratio = test_ratio
        self._max_time_index = max_time_index
        self._drop_last = drop_last
        self._skip_first_n = skip_first_n

    def prepare_data(self):

        print("Loading data for {}".format(self._name))
        self._data_arrays = load_arrays(self._name)



    def setup(self,**kwargs):



        n_train = int((1-self._test_ratio)*len(self._data_arrays))


        arrays_train = self._data_arrays[:n_train]
        arrays_test = self._data_arrays[n_train:]

        Xtrain, Ytrain = prepare_x_y_time_data( arrays_train ,
                                                 skip_steps = self._skip_steps,
                                                 store_steps_ahead = self._store_steps_ahead,
                                                 max_time_index = self._max_time_index,
                                                 skip_first_n = self._skip_first_n)

        self.train_dataset = SimDataset(Xtrain,Ytrain)


        Xtest, Ytest = prepare_x_y_time_data( arrays_test ,
                                               skip_steps = self._skip_steps,
                                               store_steps_ahead = self._store_steps_ahead,
                                               max_time_index = self._max_time_index,
                                               skip_first_n = self._skip_first_n )

        self.test_dataset = SimDataset(Xtest,Ytest)

        self.test_samples = self._select_test_samples(arrays_test, self._skip_steps)

    def get_test_samples(self):

        if not(hasattr(self,"test_samples")):
            raise(ValueError("DataModulePl setup has not been called yet"))

        test_samples = [torch.Tensor(array).to(self._device) for array in self.test_samples]

        return test_samples

    def _select_test_samples(self, data_arrays, skip_steps):

        if not(hasattr(self, "_data_arrays")):
            raise(ValueError("prepare data has not been called yet"))

        index_samples = [int(val) for val in np.linspace(0,len(data_arrays), 11)]
        index_samples[-1]=index_samples[-1]-1
        time_init = [30, 80, 60,  70, 100,  100, 90,80,  120,120, 200, 220]

        test_samples = []
        for i,index in enumerate(index_samples):
            test_samples.append(data_arrays[index][time_init[i]:][::skip_steps][:,None,:])

        test_samples = np.array(test_samples)


        return test_samples

    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size = self._batch_size, shuffle = True, drop_last = self._drop_last, num_workers = 1)

    def val_dataloader(self):

        return DataLoader(self.test_dataset, batch_size = self._batch_size, shuffle = True, drop_last = self._drop_last, num_workers = 1)




class ModelEvaluationCallback(Callback):

    def __init__(self, datamodule, save_dir,  period_evaluation_epochs = 5, on_train_start = True):
        self._period_evaluation_epochs = period_evaluation_epochs
        self._data_module = datamodule
        self._save_dir = save_dir
        self._on_train_start = on_train_start

    def on_train_start(self, trainer, model):

        if self._on_train_start:
            epoch = trainer.current_epoch
            test_samples = self._data_module.get_test_samples()
            try:
                evaluate_model(model, test_samples, save_dir = self._save_dir, preffix_name = "0_start_training_epoch_{}".format(epoch))
            except Exception as e: ###dirty fix
                print(e)
                raise(e)
                with open("./errors.txt",'w') as f:
                    f.write("epoch {}".format(epoch))
                    f.write(str(e))
        else:
            pass


    def on_train_end(self, trainer, model):
        epoch = trainer.current_epoch
        test_samples = self._data_module.get_test_samples()
        try:
            evaluate_model(model, test_samples, save_dir = self._save_dir, preffix_name = "last_training_epoch_{}".format(epoch))
        except Exception as e: ###dirty fix
            print(e)
            raise(e)
            with open("./errors.txt",'w') as f:
                f.write("epoch {}".format(epoch))
                f.write(str(e))
        else:
            pass

    def on_epoch_end(self, trainer, model):

        epoch = trainer.current_epoch
        test_samples = self._data_module.get_test_samples()

        if epoch>0:
            if (epoch%self._period_evaluation_epochs)==0:
                try:
                    evaluate_model(model, test_samples, save_dir = self._save_dir, preffix_name = "epoch_{}".format(epoch))
                except Exception as e: ###dirty fix
                    print(e)
                    with open("./errors.txt",'a+') as f:
                        f.write("epoch {}".format(epoch))
                        f.write(str(e))
       
class BaseModelPL(pl.LightningModule):

    def __init__(self, results_dir = ".", tol_next_step = 0.0015 , lr = 1e-3, weight_decay = 1e-5):
        super().__init__()

        self.criterion = torch.nn.L1Loss()
        self._results_dir = results_dir
        self._tol_next_step = tol_next_step #if error less than this adds next step
        self._n_steps_ahead = 0
        self._lr = lr
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._weight_decay = weight_decay


        try:
            os.makedirs(self._results_dir)
        except:
            pass

    def forward(self,x):

        x = self._model(x)

        return x

    def training_step(self, batch, batch_idx):



        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"],batch["Y"][:,0,...],batch["Y"][:,1,...],batch["Y"][:,2,...],batch["Y"][:,3,...]
        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = Xb.to(self._device), Ystep1.to(self._device), Ystep2.to(self._device), Ystep3.to(self._device), Ystep4.to(self._device)
        Ydata = [Ystep1, Ystep2, Ystep3, Ystep4]

        Ypred1 = self(Xb)

        loss1 = self.criterion(Ypred1, Ystep1)

        Ypred = Ypred1

        losses = []
        losses.append(loss1)

        for i in range(1,self._n_steps_ahead):


            Ypred = self(Ypred)

            losses.append(self.criterion(Ypred, Ydata[i]))

        losses = [l.view(1) for l in losses]
        loss = torch.mean( torch.cat(losses, 0) )

        self.log("train_loss_step", loss)

        return {"loss":loss, "log":{"train_loss": loss, "loss_s1": losses[0], "n_steps_ahead": self._n_steps_ahead}}

    def training_epoch_end(self, train_step_results):

        epoch_training_loss = torch.mean(torch.Tensor([d["loss"] for d in train_step_results]))

        lr = self.optimizers().param_groups[0].get("lr")

        self.log("lr", lr)
        self.log("train_loss", epoch_training_loss)


    def validation_epoch_end(self, validation_step_outputs):

        vs_outputs = [[d["vl_1"],d["vl_2"], d["vl_3"], d["vl_4"]] for d in validation_step_outputs]

        validation_step_outputs = np.array(torch.mean(torch.Tensor(vs_outputs),axis =0))

        val_loss1 = validation_step_outputs[0]
        val_loss2 = validation_step_outputs[1]
        val_loss3 = validation_step_outputs[2]
        val_loss4 = validation_step_outputs[3]

        self.log("val_loss",val_loss1)
        self.log("val_loss1", val_loss1)
        self.log("val_loss2", val_loss2)
        self.log("val_loss3", val_loss3)
        self.log("val_loss4", val_loss4)
        self.log("n_steps_ahead",self._n_steps_ahead)


        if (val_loss1 < (self._tol_next_step/(self._n_steps_ahead+1)) and self._n_steps_ahead <=2):#4 steps for now


            self._n_steps_ahead +=1

            print("advancing n steps ahead {}",self._n_steps_ahead)





    def validation_step(self, batch, batch_idx):

        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"],batch["Y"][:,0,...],batch["Y"][:,1,...],batch["Y"][:,2,...],batch["Y"][:,3,...]
        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = Xb.to(self._device), Ystep1.to(self._device), Ystep2.to(self._device), Ystep3.to(self._device), Ystep4.to(self._device)
        Ypred1 = self(Xb)
        Ypred2 = self(Ypred1)
        Ypred3 = self(Ypred2)
        Ypred4 = self(Ypred3)


        val_loss1 = self.criterion(Ypred1, Ystep1)
        val_loss2 = self.criterion(Ypred2, Ystep2)
        val_loss3 = self.criterion(Ypred3, Ystep3)
        val_loss4 = self.criterion(Ypred4, Ystep4)



        return {"vl_1":val_loss1,
                "vl_2": val_loss2,
                "vl_3": val_loss3,
                "vl_4": val_loss4}




    def test_step(self, batch, batch_idx):

        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"],batch["Y"][:,0,...],batch["Y"][:,1,...],batch["Y"][:,2,...],batch["Y"][:,3,...]
        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = Xb.to(self._device), Ystep1.to(self._device), Ystep2.to(self._device), Ystep3.to(self._device), Ystep4.to(self._device)
        Ypred1 = self(Xb)
        Ypred2 = self(Ypred1)
        Ypred3 = self(Ypred2)
        Ypred4 = self(Ypred3)

        val_loss1 = self.criterion(Ypred1, Ystep1)
        val_loss2 = self.criterion(Ypred2, Ystep2)
        val_loss3 = self.criterion(Ypred3, Ystep3)
        val_loss4 = self.criterion(Ypred4, Ystep4)




        return {"loss_s1": val_loss1, "loss_s2": val_loss2, "loss_s3": val_loss3, "loss_s4":val_loss4, "progress_bar":{"s1_val":val_loss1}}



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr = self._lr,  weight_decay= self._weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.8, threshold = 0.5*1e-3 ,verbose = True, eps = 1e-5)


        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels , n_layers = 2, stride = 1, padding = 1, normalization = True):

        super().__init__()

        layers = []

        for i in range(n_layers):

            if i == 0:
                _in_channels = in_channels
                _stride = stride
            else:
                _in_channels = out_channels
                _stride = 1

            layer = nn.Conv1d(_in_channels, out_channels, kernel_size=3, stride = _stride,
                     padding=padding, bias=True) #Bias can be set to false if using batch_norm ( is present there)

            torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

            layers.append(layer)

        if normalization:
            self.norm = torch.nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()







        self._layers = nn.ModuleList(layers)

        if (in_channels != out_channels) or (stride>1):

            self._shortcut = nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = padding)

        else:

            self._shortcut = nn.Identity()

        self._activation = torch.nn.ReLU()

    def forward(self, x):

        _x = x

        for layer in self._layers:

            _x = self._activation(layer(_x))

        out = self.norm(self._shortcut(x) + _x)#WRONG BATCH NORM WRONGLY APP

        return out


class BasicNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, blocks = [2, 2, 2, 2, 2], add_input_output = True, normalization = True):

        super().__init__()

        layers = []

        for i,_block in enumerate(blocks):


            if i == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels


            layer = ResidualBlock(_in_channels, hidden_channels, stride = 1, padding=1, normalization = normalization)

            layers.append(layer)



        self._hidden_layers = nn.ModuleList(layers)


        self._out_layer = nn.Conv1d( hidden_channels , out_channels, kernel_size=1, stride = 1,
             padding=0, bias=True)

        self._add_input_output = add_input_output


        self.act_out = torch.nn.Tanh()


    def forward(self, x):

        _x = x

        for layer in self._hidden_layers:

            _x = layer(_x)

        if self._add_input_output:

            _x = self._out_layer(_x) + x

        else:

            _x = self._out_layer(_x)

        #_x = self.act_out(_x)
        return _x


class BasicNet_pl(BaseModelPL):

    def __init__(self, in_channels, hidden_channels, out_channels,  results_dir = ".", tol_next_step = 0.0015 , lr = 1e-3, weight_decay = 1e-5):

        super().__init__(results_dir = results_dir, tol_next_step = tol_next_step , lr = lr, weight_decay = weight_decay)
        self._model = BasicNet(in_channels, hidden_channels, out_channels)
        self.save_hyperparameters()

class FNO_1d_time_pl(BaseModelPL):

    def __init__(self, modes, width, norm = True,   results_dir = ".", tol_next_step = 0.0015 , lr = 1e-3, weight_decay = 1e-5):

        super().__init__( results_dir = results_dir, tol_next_step = tol_next_step , lr = lr, weight_decay = weight_decay)

        self._model = FNO1d_time(modes, width, norm = norm)

        self.save_hyperparameters()
