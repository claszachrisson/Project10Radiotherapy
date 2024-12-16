from pathlib import Path
import numpy as np
import scipy.sparse as sp
from math import floor

import CORT

try:
    import CORT.config as config
except ImportError:
    with open("CORT/config.py", "w") as f:
        f.write("CORT_path = 'insert_path_to_CORT_directory'\nbinaries_path = 'CORT/binaries/'")
    import CORT.config as config

################################################################################

dataset = 'CORT'

def get_config(case):
    if case == 'Prostate':
        # specify the data path
        data_path = f'{config.CORT_path}/Data/Prostate'

        # gantry levels to consider
        # np.arange(0, 359, 360/5)
        gantry_angles = [0, 72, 144, 216, 288]
        couch_angles = [0, 0, 0, 0, 0]

        # structure to color map
        OBJ = {
            'PTV_68':{'COLOR':'tab:blue'},
            'PTV_56':{'COLOR':'tab:cyan'},
            'Rectum':{'COLOR':'tab:green'},
            'BODY':{'COLOR':'black'},
            'Bladder':{'COLOR':'tab:orange'},
            'Penile_bulb':{'COLOR':'tab:red'},
            'Lt_femoral_head':{'COLOR':'tab:pink'},
            'Rt_femoral_head':{'COLOR':'tab:purple'},
            'Lymph_Nodes':{'COLOR':'tab:brown'},
            'prostate_bed':{'COLOR':'tab:olive'}
        }

        PTV_structure = 'PTV_68'
        PTV_dose = 68.0

        BODY_structure = 'BODY'
        BODY_threshold = 5.0

        OAR_structures = ['Rectum','Bladder','Penile_bulb',
                          'Lt_femoral_head','Rt_femoral_head',
                          'Lymph_Nodes']
        OAR_threshold = 5.0

        dim = np.array([184, 184, 90])
        dim = np.roll(dim, 1)

        # eta = 87500.0
        eta = -1.0
        steps = 20

        return data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BODY_threshold, OAR_structures, OAR_threshold, eta, steps


    elif case == 'Liver':
        # specify the data path
        data_path = f'{config.CORT_path}/Liver/'

        # gantry levels to consider
        gantry_angles = [32, 90, 148, 212, 270, 328]
        couch_angles = [0, 0, 0, 0, 0, 0]

        # structure to color map
        OBJ = {
            'CTV':{'COLOR':'tab:olive'},
            'Celiac':{'COLOR':'tab:olive'},
            'DoseFalloff':{'COLOR':'tab:olive'},
            'GTV':{'COLOR':'tab:olive'},
            'LargeBowel':{'COLOR':'tab:olive'},
            'SmallBowel':{'COLOR':'tab:olive'},
            'SMASMV':{'COLOR':'tab:olive'},
            'duodenum':{'COLOR':'tab:olive'},
            'entrance':{'COLOR':'tab:olive'},
            'PTV':{'COLOR':'tab:blue'},
            'Heart':{'COLOR':'tab:green'},
            'Skin':{'COLOR':'black'},
            'Liver':{'COLOR':'tab:orange'},
            'SpinalCord':{'COLOR':'tab:red'},
            'KidneyL':{'COLOR':'tab:pink'},
            'KidneyR':{'COLOR':'tab:purple'},
            'Stomach':{'COLOR':'tab:brown'}
        }

        PTV_structure = 'PTV'
        PTV_dose = 56.0 # Seb made that up

        BODY_structure = 'Skin'
        BODY_threshold = 5.0

        OAR_structures = ['Heart','Liver','SpinalCord',
                          #'KidneyL','KidneyR',
                          'Stomach']
        OAR_threshold = 5.0

        dim = np.array([217, 217, 168])
        dim = np.roll(dim, 1)

        # eta = 65000.0
        eta = -1.0
        steps = 20

        return data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BODY_threshold, OAR_structures, OAR_threshold, eta, steps


    elif case == 'HeadAndNeck':
        # specify the data path
        data_path = f'{config.CORT_path}/HeadAndNeck/'

        # gantry levels to consider
        # np.arange(0, 359, 360/5)
        # gantry_angles = [0, 72, 144, 216, 288]
        # np.arange(0, 359, int(360/8)+1)
        gantry_angles = [0, 52, 104, 156, 208, 260, 312]
        couch_angles = [0, 0, 0, 0, 0, 0, 0]

        # structure to color map
        OBJ = {
            'CEREBELLUM':{'COLOR':'tab:olive'},
            'CTV56':{'COLOR':'tab:olive'},
            'CTV63':{'COLOR':'tab:olive'},
            'GTV':{'COLOR':'tab:olive'},
            'LARYNX':{'COLOR':'tab:brown'},
            'LENS_LT':{'COLOR':'tab:olive'},
            'LENS_RT':{'COLOR':'tab:olive'},
            'LIPS':{'COLOR':'tab:cyan'},
            'OPTIC_NRV_LT':{'COLOR':'tab:olive'},
            'OPTIC_NRV_RT':{'COLOR':'tab:olive'},
            'TEMP_LOBE_LT':{'COLOR':'tab:olive'},
            'TEMP_LOBE_RT':{'COLOR':'tab:olive'},
            #'TM_JOINT_LT':{'COLOR':'tab:olive'},
            #'TM_JOINT_RT':{'COLOR':'tab:olive'},
            'PTV56':{'COLOR':'tab:olive'},
            'PTV63':{'COLOR':'tab:olive'},
            'PTV70':{'COLOR':'tab:blue'},
            'BRAIN_STEM_PRV':{'COLOR':'tab:olive'},
            'BRAIN_STEM':{'COLOR':'tab:green'},
            'External':{'COLOR':'black'},
            'CHIASMA':{'COLOR':'tab:orange'},
            'SPINAL_CORD':{'COLOR':'tab:red'},
            'SPINL_CRD_PRV':{'COLOR':'tab:olive'},
            'PAROTID_LT':{'COLOR':'tab:pink'},
            'PAROTID_RT':{'COLOR':'tab:purple'}
        }

        PTV_structure = 'PTV70'
        PTV_dose = 70.0

        BODY_structure = 'External'
        BODY_threshold = 5.0

        OAR_structures = ['BRAIN_STEM','CHIASMA','SPINAL_CORD',
                          'PAROTID_LT','PAROTID_RT',
                          'LARYNX', 'LIPS']
        OAR_threshold = 5.0

        dim = np.array([160, 160, 67])
        dim = np.roll(dim, 1)

        # eta = 11.0
        # eta = 3.7430655731621476
        eta = -1.0
        steps = 20

        return data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BODY_threshold, OAR_structures, OAR_threshold, eta, steps

    else:
        raise NotImplementedError


def get_D_matrices(case='Prostate', save_to_files=False):

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
    
    cfg = get_config(case)

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
    D = sp.vstack((D_full[BDY_indices],
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
        sp.save_npz(config.binaries_path + case + '_D_BDY.npz', D_BDY)
        sp.save_npz(config.binaries_path + case + '_D_OAR.npz', D_OAR)
        sp.save_npz(config.binaries_path + case + '_D_PTV.npz', D_PTV)

        np.save(config.binaries_path + case + '_target_doze_PTV.npy', target_dose_PTV)
    
    return (D_BDY, D_OAR, D_PTV, n_BDY, n_OAR, n_PTV, BDY_threshold, OAR_threshold, PTV_dose)

def prob(case='Prostate', from_files=False, BDY_downsample=1, OAR_downsample=1, PTV_downsample=1):

    # x = (t ybdy+ ybdy- yoar+ yoar- yptv+ yptv-).T

    # Relevant indices are picked out from D_full, hstack'd for each couch/gantry angle pair
    # Meaning shape[1] of D_XXX is sum of beamlets for each angle pair
    if(not from_files):
        (D_BDY, D_OAR, D_PTV, n_BDY, n_OAR, n_PTV, BDY_threshold, OAR_threshold, PTV_dose) = get_D_matrices(case, False)
        D_BDY = sp.bsr_array(D_BDY)
        D_OAR = sp.bsr_array(D_OAR)
        D_PTV = sp.bsr_array(D_PTV)
    else:
        # load data in binary form
        D_BDY = sp.load_npz(config.binaries_path + case + '_D_BDY.npz')
        D_OAR = sp.load_npz(config.binaries_path + case + '_D_OAR.npz')
        D_PTV = sp.load_npz(config.binaries_path + case + '_D_PTV.npz')
        n_BDY = D_BDY.shape[0]
        n_OAR = D_OAR.shape[0]
        n_PTV = D_PTV.shape[0]

        _, _, _, _, _, PTV_dose, _, BDY_threshold, _, OAR_threshold, _, _ = get_config(case)

        #target_dose_PTV = np.load(case + '_target_doze_PTV.npy')



    len_t = D_BDY.shape[1]
    sub_n_BDY = floor(n_BDY / BDY_downsample)
    sub_n_OAR = floor(n_OAR / OAR_downsample)
    sub_n_PTV = floor(n_PTV / PTV_downsample)
    print('Downsampling to:')
    print(sub_n_BDY, sub_n_OAR, sub_n_PTV)

    np.random.seed(42)
    sub_BDY_ind = np.random.choice(n_BDY, sub_n_BDY, replace=False)
    sub_OAR_ind = np.random.choice(n_OAR, sub_n_OAR, replace=False)
    sub_PTV_ind = np.random.choice(n_PTV, sub_n_PTV, replace=False)

    sub_D_BDY = D_BDY[sub_BDY_ind].toarray()
    sub_D_OAR = D_OAR[sub_OAR_ind].toarray()
    sub_D_PTV = D_PTV[sub_PTV_ind].toarray()
    print("Downsampling done. Constructing A matrix...")

    I_BDY = np.eye(sub_n_BDY)
    I_OAR = np.eye(sub_n_OAR)
    I_PTV = np.eye(sub_n_PTV)

    A_DIM = np.vstack([sub_D_BDY, sub_D_OAR, sub_D_PTV]) # Dose influence matrices
    A_BDY = np.vstack([I_BDY, np.zeros((sub_n_OAR+sub_n_PTV, sub_n_BDY))])
    A_OAR = np.vstack([np.zeros((sub_n_BDY, sub_n_OAR)), I_OAR, np.zeros((sub_n_PTV, sub_n_OAR))])
    A_PTV = np.vstack([np.zeros((sub_n_BDY+sub_n_OAR, sub_n_PTV)), I_PTV])

    A = np.hstack([A_DIM, -A_BDY, A_BDY, -A_OAR, A_OAR, -A_PTV, A_PTV])
    
    print("A matrix done. Constructing B vector...")
    b = np.vstack([np.full((sub_n_BDY, 1), BDY_threshold), 
                   np.full((sub_n_OAR,1), OAR_threshold), 
                   np.full((sub_n_PTV,1), PTV_dose)]).flatten()

    print("b vector done. Constructing C matrix...")
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

    i = list(np.hstack([np.arange((len_t+sub_n_BDY), (len_t+2*sub_n_BDY)),
                  np.arange((len_t+2*sub_n_BDY+sub_n_OAR), (len_t+2*sub_n_BDY+2*sub_n_OAR)),
                  np.arange((len_t+2*sub_n_BDY+2*sub_n_OAR+sub_n_PTV), (len_t+2*sub_n_BDY+2*sub_n_OAR+2*sub_n_PTV))]).flatten())
    print("Problem construction done.")
    print(f"Number of basic variables: {len(b)}")
    print(f"Number of non-basic variables: {len(C[0,:])-len(b)}")
    return A,b,C,i