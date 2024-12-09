import fire

from time import time
from os.path import exists

import numpy as np
import scipy as sp
import scipy.sparse

import CORT
import config
# import utils

################################################################################


def main(case='Prostate', loss='squared', score_method='gradnorm', m=25000, repetitions=1, w_BDY_over=1.0, w_OAR_over=1.0, w_PTV_over=4096.0, w_PTV_under=4096.0):

    """
    case = 'Prostate'
    case = 'Liver'
    case = 'HeadAndNeck'

    loss = 'absolute'
    loss = 'squared'

    score_method = 'gradnorm'
    m = 25000
    repetitions = 1
    w_BDY_over = 1.0
    w_OAR_over = 1.0
    w_PTV_over = 4096.0
    w_PTV_under = 4096.0
    """


    if loss not in ['absolute','squared']:
        exit(f'ERROR: loss is {loss} but it should be either absolute, or squared')

    if case not in ['Prostate','Liver','HeadAndNeck']:
        exit(f'ERROR: case is {case} but it should be either Prostate, Liver, or HeadAndNeck')

    if score_method not in ['full', 'uniform', 'gradnorm', 'residual']:
        # full is a dummy and is not used, m=0 indicates full
        exit(f'ERROR: score_method is {score_method} but it should be either uniform, gradnorm, or residual')


    m = int(m)
    repetitions = int(repetitions)
    w_BDY_over = float(w_BDY_over)
    w_OAR_over = float(w_OAR_over)
    w_PTV_over = float(w_PTV_over)
    w_PTV_under = float(w_PTV_under)

    # file name format is ...
    filename = f'result_{case}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
    if loss == 'squared':
        filename = f'result_{case}_{loss}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'

    # dont run the script if results file already exists!
    if exists(filename):
        print(f'ABORT: experiment {filename} already exists!')
        exit()

    cfg = config.get_config(case)
    data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BDY_threshold, OAR_structures, OAR_threshold, eta, steps = cfg

    configuraton = {
        'case':case,
        'loss':loss,
        'score_method':score_method,
        'm':m,
        'repetitions':repetitions,
        'w_BDY_over':w_BDY_over,
        'w_OAR_over':w_OAR_over,
        'w_PTV_over':w_PTV_over,
        'w_PTV_under':w_PTV_under,
        'start_time':time()
    }
    print('configuraton:', configuraton)
    # save at least the config such that the file exists and no parallel run of
    # the same experiment can happen
    np.savez(filename, configuraton=configuraton)


    # load full dose influence matrix
    D_full = CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))


    # set the indices for body (BDY), OAR, and PTV
    BDY_indices = OBJ[BODY_structure]['IDX']
    PTV_indices = OBJ[PTV_structure]['IDX']
    OAR_indices = np.unique(np.hstack([OBJ[OAR_structure]['IDX'] for OAR_structure in OAR_structures]))
    # fix the indices
    OAR_indices = np.setdiff1d(OAR_indices, PTV_indices)
    BDY_indices = np.setdiff1d(BDY_indices, np.union1d(PTV_indices, OAR_indices))

    assert len(np.intersect1d(BDY_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, BDY_indices)) == 0

    n_BDY = len(BDY_indices)
    n_OAR = len(OAR_indices)
    n_PTV = len(PTV_indices)

    # specify the target dose
    # initialize the target dose to zero
    target_dose = np.zeros(D_full.shape[0])
    # set the PTV dose
    target_dose[OBJ[PTV_structure]['IDX']] = PTV_dose
    # set the OAR target dose to a threshold to prevent penalizing small violations
    target_dose[OAR_indices] = OAR_threshold
    # set the BDY target dose to a threshold to prevent penalizing small violations
    target_dose[BDY_indices] = BDY_threshold


    # set D and overwrite target_dose to only consider BODY, OAR, and PTV voxels,
    # i.e., skip all other voxels outside the actual BODY
    D = sp.sparse.vstack((D_full[BDY_indices],
                          D_full[OAR_indices],
                          D_full[PTV_indices]))
    target_dose = np.hstack((target_dose[BDY_indices],
                             target_dose[OAR_indices],
                             target_dose[PTV_indices]))


    D_BDY = D[:n_BDY]
    D_OAR = D[n_BDY:(n_BDY+n_OAR)]
    D_PTV = D[(n_BDY+n_OAR):]
    target_dose_PTV = target_dose[(n_BDY+n_OAR):]

    # Save data in binary form
    scipy.sparse.save_npz('D_BDY.npz', D_BDY)
    scipy.sparse.save_npz('D_OAR.npz', D_OAR)
    scipy.sparse.save_npz('D_PTV.npz', D_PTV)

    np.save('taget_doze_PTV.npy', target_dose_PTV)


if __name__ == '__main__':
  fire.Fire(main)






