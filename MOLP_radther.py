import numpy as np
from scipy.sparse import bsr_array, vstack,load_npz
from get_matrices import main as get_matrices
import CORT.config as config
from math import floor

def prob(case='Prostate', from_files=False, BDY_downsample=1, OAR_downsample=1, PTV_downsample=1):

    # x = (t ybdy+ ybdy- yoar+ yoar- yptv+ yptv-).T

    # Relevant indices are picked out from D_full, hstack'd for each couch/gantry angle pair
    # Meaning shape[1] of D_XXX is sum of beamlets for each angle pair
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



    len_t = D_BDY.shape[1]
    sub_n_BDY = floor(n_BDY / BDY_downsample)
    sub_n_OAR = floor(n_OAR / OAR_downsample)
    sub_n_PTV = floor(n_PTV / PTV_downsample)
    print('Downsampling to:')
    print(sub_n_BDY, sub_n_OAR, sub_n_PTV)

    sub_BDY_ind = np.random.choice(n_BDY, sub_n_BDY, replace=False)
    sub_OAR_ind = np.random.choice(n_OAR, sub_n_OAR, replace=False)
    sub_PTV_ind = np.random.choice(n_PTV, sub_n_PTV, replace=False)

    sub_D_BDY = D_BDY[sub_BDY_ind].toarray()
    sub_D_OAR = D_OAR[sub_OAR_ind].toarray()
    sub_D_PTV = D_PTV[sub_PTV_ind].toarray()

    I_BDY = np.eye(sub_n_BDY)
    I_OAR = np.eye(sub_n_OAR)
    I_PTV = np.eye(sub_n_PTV)

    A_DIM = np.vstack([sub_D_BDY, sub_D_OAR, sub_D_PTV]) # Dose influence matrices
    A_BDY_POS = np.vstack([I_BDY, np.zeros((sub_n_OAR+sub_n_PTV, sub_n_BDY))])
    A_BDY_NEG = -A_BDY_POS
    A_OAR_POS = np.vstack([np.zeros((sub_n_BDY, sub_n_OAR)), I_OAR, np.zeros((sub_n_PTV, sub_n_OAR))])
    A_OAR_NEG = -A_OAR_POS
    A_PTV_POS = np.vstack([np.zeros((sub_n_BDY+sub_n_OAR, sub_n_PTV)), I_PTV])
    A_PTV_NEG = -A_PTV_POS

    A = np.hstack([A_DIM, A_BDY_NEG, A_BDY_POS, A_OAR_NEG, A_OAR_POS, A_PTV_NEG, A_PTV_POS])

    b = np.vstack([np.full((sub_n_BDY, 1), BDY_threshold), 
                   np.full((sub_n_OAR,1), OAR_threshold), 
                   np.full((sub_n_PTV,1), PTV_dose)])

    Ct = np.zeros((3,len_t))
    CBDY1 = np.vstack([ np.ones((1, sub_n_BDY)),
                     np.zeros((2, sub_n_BDY)) ])
    CBDY0 = np.zeros((3,sub_n_BDY))
    COAR1 = np.vstack([ np.zeros((1,sub_n_OAR)),
                     np.ones((1, sub_n_OAR)),
                     np.zeros((1, sub_n_OAR)) ])
    COAR0 = np.zeros((3,sub_n_OAR))
    CPTV1 = np.vstack([ np.zeros((2,sub_n_PTV)),
                     np.ones((1, sub_n_PTV)) ])
    CPTV0 = np.zeros((3,sub_n_PTV))

    C = np.hstack([Ct, CBDY1, CBDY0, COAR1, COAR0, CPTV0, CPTV1])

    return A,b,C

prob('Liver', True, 100, 100, 100)