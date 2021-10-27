import os
from solvers import solve_allen_cahn_1D
import fire
from utils import post_process_save_pde_1d
from config import settings as stts

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





if __name__ == "__main__":
    fire.Fire(RunSimulations)
