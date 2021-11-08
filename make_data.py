import os
from solvers import solve_allen_cahn_1D,solve_burgers_1D
import fire
from config import settings as stts
import numpy as np
from utils import *


class RunSimulations():

    def __init__(self, results_dir = None):

        if not(results_dir):
            self._results_dir = stts["dir_data"]
        else:
            self._results_dir = results_dir

    def solve_allen_cahn_1d(self, nsamples, tc = 1, xc = 0.1, eps = 1e-2, T = 60 , initial_conditions = "random"):

        save_dir = os.path.join(self._results_dir,"AC1d")

        try:
            os.makedirs(save_dir)
        except:
            pass

        for sample in range(nsamples):

            name_simulation = initial_conditions+"{}_tc_{}_xc_{}_eps_{}_T_{}".format(sample,
                                                                      str(tc).replace(".","p"),
                                                                      str(xc).replace(".","p"),
                                                                                    str(eps).replace(".","p"),
                                                                                    str(T).replace(".","p"))

            sol = solve_allen_cahn_1D(tc = tc, xc = xc, eps = eps, T = T ,
                                       initial_conditions = initial_conditions)


            post_process_save_pde_1d(sol, name_simulation, save_dir)

    def solve_burgers_1d(self, nu = 0.5*1e-1, T = 2.0, initial_condition_n = 1,n_elements = 100, n_steps = 100, preffix = ""):
        save_dir = os.path.join(self._results_dir,"Burgers1d")

        try:
            os.makedirs(save_dir)
        except Exception as e:
            print(e)



        name_simulation = "{}_initial_condition_{}_nu_{}_T_{}".format(preffix, initial_condition_n,
                                                                    str(nu).replace(".","p"),
                                                                     str(T).replace(".","p")
                                                                    )
        sol = solve_burgers_1D(nu = nu, T = T, initial_condition_n = initial_condition_n, n_elements = n_elements, n_steps = n_steps )


        post_process_save_pde_1d(sol, name_simulation, save_dir)

    def solve_multiple_burgers(self):

        nus = np.linspace(0.1,0.001,20)

        for i in range(len(nus)):
            self.solve_burgers_1d(nu = nus[i], initial_condition_n = 1, preffix = i)
        for i in range(len(nus)):
            self.solve_burgers_1d(nu = nus[i], initial_condition_n = 2, preffix = i)





if __name__ == "__main__":
    fire.Fire(RunSimulations)
