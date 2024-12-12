import numpy as np
from scipy.sparse import bsr_array, vstack,load_npz
from get_matrices import main as get_matrices
import CORT.config as config

def prob(case='Prostate', from_files=False):
    n_obj = 3

    # Relevant indices are picked out from D_full, hstack'd for each couch/gantry angle pair
    # Meaning shape[1] (and consequently, length of x in MOLP) of D_XXX is sum of beamlets for each angle pair
    if(not from_files):
        (D_BDY, D_OAR, D_PTV, n_BDY, n_OAR, n_PTV, BDY_threshold, OAR_threshold, PTV_dose) = get_matrices(case, False)
        D_BDY = bsr_array(D_BDY)
        D_OAR = bsr_array(D_OAR)
        D_PTV = bsr_array(D_PTV)
    else:
        # load data in binary form
        D_BDY = load_npz(config.binaries_path + case + '_D_BDY.npz')
        D_OAR = load_npz(config.binaries_path + case + '_D_OAR.npz')
        D_PTV = load_npz(config.binaries_path + case + '_D_PTV.npz')
        n_BDY = D_BDY.shape[0]
        n_OAR = D_OAR.shape[0]
        n_PTV = D_PTV.shape[0]

        _, _, _, _, _, PTV_dose, _, BDY_threshold, _, OAR_threshold, _, _ = config.get_config(case)

        #target_dose_PTV = np.load(case + '_target_doze_PTV.npy')


    C = np.zeros((n_obj, D_BDY.shape[1]))

    # C matrix becomes column sums of D matrices, divided by no. of voxels in structure
    # One row in Cx is thus average of sum of dose in structure
    C[0] = np.sum(D_BDY, axis=0) / n_BDY
    C[1] = np.sum(D_OAR, axis=0) / n_OAR
    C[2] = np.sum(D_PTV, axis=0) / n_PTV

    A = vstack([D_BDY,D_OAR,D_PTV])

    b = np.vstack([np.full((n_BDY, 1), BDY_threshold), 
                  np.full((n_OAR,1), OAR_threshold), 
                  np.full((n_PTV,1), -PTV_dose)])

    
    return (A,b,C)

prob('Liver', True)