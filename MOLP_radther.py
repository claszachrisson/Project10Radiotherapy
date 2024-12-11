import numpy as np
from scipy.sparse import bsr_array, vstack
from get_matrices import main as get_matrices

# edit path to CORT data as necessary
#CORT_path = '.'
CORT_path = '/Users/claszachrisson/Downloads/MOLP/efficient_rtp'

def prob(case='Prostate'):
    n_obj = 3

    # Relevant indices are picked out from D_full, hstack'd for each couch/gantry angle pair
    # Meaning shape[1] (and consequently, length of x in MOLP) of D_XXX is sum of beamlets for each angle pair
    (D_BDY, D_OAR, D_PTV, n_BDY, n_OAR, n_PTV, BDY_threshold, OAR_threshold, target_dose_PTV) = get_matrices(case, False, CORT_path)

    D_BDY = bsr_array(D_BDY)
    D_OAR = bsr_array(D_OAR)
    D_PTV = bsr_array(D_PTV)
    print(D_BDY.shape)

    C = np.zeros((n_obj, D_BDY.shape[1]))

    # C matrix becomes column sums of D matrices, divided by no. of voxels in structure
    # One row in Cx is thus average of sum of dose in structure
    C[0] = np.sum(D_BDY, axis=0) / n_BDY
    C[1] = np.sum(D_OAR, axis=0) / n_OAR
    C[2] = np.sum(D_PTV, axis=0) / n_PTV

    A = vstack(D_BDY,D_OAR,D_PTV)
    b = np.vstack(BDY_threshold * np.ones(n_BDY, 1), 
                  OAR_threshold * np.ones(n_OAR,1), 
                  -target_dose_PTV * np.ones(n_PTV,1))
    
    return (A,b,C)

prob('Liver')