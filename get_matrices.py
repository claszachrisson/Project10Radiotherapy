import fire

from pathlib import Path
import numpy as np
import scipy as sp
import scipy.sparse

import CORT.CORT as CORT
import CORT.config as config
# import utils

################################################################################

def main(case='Prostate', save_to_files=False):

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

    if case not in ['Prostate','Liver','HeadAndNeck']:
        exit(f'ERROR: case is {case} but it should be either Prostate, Liver, or HeadAndNeck')
    
    cfg = config.get_config(case)

    data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BDY_threshold, OAR_structures, OAR_threshold, eta, steps = cfg


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

    if(save_to_files):
        # Save data in binary form
        Path(config.binaries_path).mkdir(parents=True, exist_ok=True)
        scipy.sparse.save_npz(config.binaries_path + case + '_D_BDY.npz', D_BDY)
        scipy.sparse.save_npz(config.binaries_path + case + '_D_OAR.npz', D_OAR)
        scipy.sparse.save_npz(config.binaries_path + case + '_D_PTV.npz', D_PTV)

        np.save(config.binaries_path + case + '_target_doze_PTV.npy', target_dose_PTV)
    
    return (D_BDY, D_OAR, D_PTV, n_BDY, n_OAR, n_PTV, BDY_threshold, OAR_threshold, PTV_dose)


if __name__ == '__main__':
  fire.Fire(main)