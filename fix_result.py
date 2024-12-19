import numpy as np
import CORT.utils as utils


A,b,C, i = utils.prob('Liver', True, BDY_downsample=10000, OAR_downsample=1000, PTV_downsample=100)
num_variables = len(C[0,:])
result = np.load('result.npz') #Need to covert solution_vec to solutions in the now saved file



eff_ind = result['list_data']  
solution_vec = result['array_data']

solutions = np.zeros((len(eff_ind),num_variables))
for i,sol in enumerate(solutions):
    sol[eff_ind[i]]=solution_vec[i]

np.savez(f'result_liver_BDY_downsample_{10000}_OAR_downsample_{1000}_PTV_downsample{100}.npz', list_data=eff_ind, array_data=solutions)

