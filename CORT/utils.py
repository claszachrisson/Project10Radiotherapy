from pathlib import Path
import numpy as np
import scipy.sparse as sp
from math import floor

import CORT.CORT as CORT

################################################################################

dataset = 'CORT'

class Config():
    def __init__(self, case):
        self.case = case
        self.data_path = f'CORT'
        self.filenames = ""
        self.gantry_angles = []
        self.couch_angles = []
        self.OBJ = {}
        self.PTV_structure = ""
        self.PTV_dose = 0
        self.BODY_structure = ""
        self.BODY_threshold = 0
        self.OAR_structures = []
        self.OAR_threshold = 0
        self.dim = []
        self.n_vars = 0

def get_config(case,filenames=None):
    cfg = Config(case)
    match case:
        case 'Prostate':
            # gantry levels to consider
            # np.arange(0, 359, 360/5)
            cfg.gantry_angles = [0, 72, 144, 216, 288]
            cfg.couch_angles = [0, 0, 0, 0, 0]

            # structure to color map
            cfg.OBJ = {
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

            cfg.PTV_structure = 'PTV_68'
            cfg.PTV_dose = 68.0

            cfg.BODY_structure = 'BODY'
            cfg.BODY_threshold = 5.0

            cfg.OAR_structures = ['Rectum','Bladder','Penile_bulb',
                            'Lt_femoral_head','Rt_femoral_head',
                            'Lymph_Nodes']
            cfg.OAR_threshold = 5.0

            cfg.dim = np.array([184, 184, 90])
            cfg.n_vars = 721
        case 'Liver':

            # gantry levels to consider
            cfg.gantry_angles = [32, 90, 148, 212, 270, 328]
            cfg.couch_angles = [0, 0, 0, 0, 0, 0]

            # structure to color map
            cfg.OBJ = {
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

            cfg.PTV_structure = 'PTV'
            cfg.PTV_dose = 56.0 # Seb made that up

            cfg.BODY_structure = 'Skin'
            cfg.BODY_threshold = 5.0

            cfg.OAR_structures = ['Heart','Liver','SpinalCord',
                            #'KidneyL','KidneyR',
                            'Stomach']
            cfg.OAR_threshold = 5.0

            cfg.dim = np.array([217, 217, 168])
            cfg.n_vars = 389
        case 'HeadAndNeck':

            # gantry levels to consider
            # np.arange(0, 359, 360/5)
            # gantry_angles = [0, 72, 144, 216, 288]
            # np.arange(0, 359, int(360/8)+1)
            cfg.gantry_angles = [0, 52, 104, 156, 208, 260, 312]
            cfg.couch_angles = [0, 0, 0, 0, 0, 0, 0]

            # structure to color map
            cfg.OBJ = {
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

            cfg.PTV_structure = 'PTV70'
            cfg.PTV_dose = 70.0

            cfg.BODY_structure = 'External'
            cfg.BODY_threshold = 5.0

            cfg.OAR_structures = ['BRAIN_STEM','CHIASMA','SPINAL_CORD',
                            'PAROTID_LT','PAROTID_RT',
                            'LARYNX', 'LIPS']
            cfg.OAR_threshold = 5.0

            cfg.dim = np.array([160, 160, 67])
            #cfg.n_vars = -1
        case _:
            raise NotImplementedError
    cfg.dim = np.roll(cfg.dim, 1)
    cfg.n_voxels = np.prod(cfg.dim)
    CORT.load_indices(cfg)

    if filenames:
        cfg.filenames = filenames
    else:
        cfg.filenames = cfg.case

    return cfg

def save_D_full(case, filepath):
    cfg = get_config(case)

    # load full dose influence matrix
    D_full = CORT.get_D_full(cfg, filepath)

    sp.save_npz(f'{cfg.data_path}/binaries/{cfg.case}_D_full.npz', D_full)

def get_D_matrices(case='Prostate', save_to_files=False): # Old

    """
    case = 'Prostate'
    case = 'Liver'
    case = 'HeadAndNeck'
    """

    if case not in ['Prostate','Liver','HeadAndNeck']:
        exit(f'ERROR: case is {case} but it should be either Prostate, Liver, or HeadAndNeck')
    
    cfg = get_config(case)

    # load full dose influence matrix
    D_full = CORT.load_D_full(cfg)
    BDY_indices, OAR_indices, PTV_indices, n_BDY, n_OAR, n_PTV = get_diff_indices(cfg, True)

    # set D and overwrite target_dose to only consider BODY, OAR, and PTV voxels,
    # i.e., skip all other voxels outside the actual BODY
    D = sp.vstack((D_full[BDY_indices],
                          D_full[OAR_indices],
                          D_full[PTV_indices]))


    D_BDY = D[:n_BDY]
    D_OAR = D[n_BDY:(n_BDY+n_OAR)]
    D_PTV = D[(n_BDY+n_OAR):]

    if(save_to_files):
        # Save data in binary form
        Path('CORT/binaries/').mkdir(parents=True, exist_ok=True)
        sp.save_npz('CORT/binaries/' + case + '_D_full.npz', D_full)
        sp.save_npz('CORT/binaries/' + case + '_D_BDY.npz', D_BDY)
        sp.save_npz('CORT/binaries/' + case + '_D_OAR.npz', D_OAR)
        sp.save_npz('CORT/binaries/' + case + '_D_PTV.npz', D_PTV)
    
    #return (D_BDY, D_OAR, D_PTV, n_BDY, n_OAR, n_PTV, BODY_threshold, OAR_threshold, PTV_dose)

def get_diff_indices(cfg, lengths = False):
    # set the indices for body (BDY), OAR, and PTV
    BDY_indices = cfg.OBJ[cfg.BODY_structure]['IDX']
    PTV_indices = cfg.OBJ[cfg.PTV_structure]['IDX']
    OAR_indices = np.unique(np.hstack([cfg.OBJ[OAR_structure]['IDX'] for OAR_structure in cfg.OAR_structures]))
    # fix the indices
    OAR_indices = np.setdiff1d(OAR_indices, PTV_indices)
    BDY_indices = np.setdiff1d(BDY_indices, np.union1d(PTV_indices, OAR_indices))

    assert len(np.intersect1d(BDY_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, BDY_indices)) == 0
    if lengths:
        return BDY_indices, OAR_indices, PTV_indices, len(BDY_indices), len(OAR_indices), len(PTV_indices)
    return BDY_indices, OAR_indices, PTV_indices

def prob(case='Prostate', binary_filenames=None, BDY_downsample=1, OAR_downsample=1, PTV_downsample=1):

    # x = (t ybdy+ ybdy- yoar+ yoar- yptv+ yptv-).T

    # Relevant indices are picked out from D_full, hstack'd for each couch/gantry angle pair
    # Meaning shape[1] of D_XXX is sum of beamlets for each angle pair
    # if(not from_files):
    #     (D_BDY, D_OAR, D_PTV, n_BDY, n_OAR, n_PTV, BODY_threshold, OAR_threshold, PTV_dose) = get_D_matrices(case, False)
    #     D_BDY = sp.bsr_array(D_BDY)
    #     D_OAR = sp.bsr_array(D_OAR)
    #     D_PTV = sp.bsr_array(D_PTV)
    # else:
    #     # load data in binary form
    #     D_BDY = sp.load_npz('CORT/binaries/' + case + '_D_BDY.npz')
    #     D_OAR = sp.load_npz('CORT/binaries/' + case + '_D_OAR.npz')
    #     D_PTV = sp.load_npz('CORT/binaries/' + case + '_D_PTV.npz')
    #     n_BDY = D_BDY.shape[0]
    #     n_OAR = D_OAR.shape[0]
    #     n_PTV = D_PTV.shape[0]

    #     _, _, _, _, _, PTV_dose, _, BODY_threshold, _, OAR_threshold = get_config(case)

    #     #target_dose_PTV = np.load(case + '_target_doze_PTV.npy')

    cfg = get_config(case, binary_filenames)
    CORT.load_indices(cfg)

    #D_BDY, D_OAR, D_PTV = CORT.load_D_XYZ(cfg)
    D_full = CORT.load_D_full(cfg)
    BDY_indices, OAR_indices, PTV_indices, n_BDY, n_OAR, n_PTV = get_diff_indices(cfg, lengths=True)
    D_BDY = D_full[BDY_indices]
    D_OAR = D_full[OAR_indices]
    D_PTV = D_full[PTV_indices]

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
    b = np.vstack([np.full((sub_n_BDY, 1), cfg.BODY_threshold), 
                   np.full((sub_n_OAR,1), cfg.OAR_threshold), 
                   np.full((sub_n_PTV,1), cfg.PTV_dose)]).flatten()

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


def compute_subset(D, weights, m, seed, scores):

    assert D.shape[0] == weights.shape[0]
    assert D.shape[0] == scores.shape[0]

    n = D.shape[0]

    # sampling probabilities
    q = scores / scores.sum()

    np.random.seed(seed)
    sample_idx = np.random.choice(n, size=m, p=q)

    """
    The re-weighting is done as follows:
        new_weight = old_weight / (m*q)
    """
    new_weights  = weights / (m*q)

    # we need to return the indices and not the smaller arrays
    # because we need to split the structures
    return sample_idx, new_weights



def compute_scores(D, target_dose, weights, eta, steps):

    if eta == -1.0:
        print('compute_scores(): No eta given, computing learning rate...')
        A = sp.diags(np.sqrt(weights)) * D
        _,s,_ = sp.linalg.svds(2*A.T@A)
        # L = s.max()
        eta = 2/(s.min() + s.max())

    x_hist = []
    loss_hist = []
    score_residual_hist = []
    score_gradnorm_hist = []

    print(f'Running PGD for {steps} steps with a learning rate of eta={eta}')


    # initialize x with zeros
    x = np.zeros(D.shape[1])
    for i in range(steps):
        # compute the current doses
        dose = D@x
        # compute the residual
        res = dose - target_dose
        # use square loss and the weighing
        loss = np.sum(np.multiply(weights, res**2))
        loss_hist.append(loss)
        print(f'loss={loss} max_dose={dose.max()}')
        # compute the gradient per data point
        grad_per_point = D.multiply(2*np.multiply(weights, res).reshape(-1, 1))
        grad = np.array(grad_per_point.sum(0)).reshape(-1)
        # gradient descent step
        x = x - eta*grad
        # projection to x>=0
        x[x<0] = 0
        # remember x
        x_hist.append(x)
        # compute the scores
        score_residual = np.abs(res)
        score_residual_hist.append(score_residual)
        score_gradnorm = sp.linalg.norm(grad_per_point, ord=2, axis=1)
        score_gradnorm_hist.append(score_gradnorm)

    time_PGD = 0

    return x_hist, loss_hist, score_residual_hist, score_gradnorm_hist, time_PGD

def importance_sample_D_XYZ(case='Prostate', loss='squared', score_method='gradnorm', m=25000, repetitions=1, w_BDY_over=1.0, w_OAR_over=1.0, w_PTV_over=4096.0, w_PTV_under=4096.0):

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

    cfg = get_config(case)

    # load full dose influence matrix
    D_full = CORT.load_D_full(cfg)
    #CORT.load_indices(cfg)

    BDY_indices, OAR_indices, PTV_indices, n_BDY, n_OAR, n_PTV = get_diff_indices(cfg, True)

    # specify the target dose
    # initialize the target dose to zero
    target_dose = np.zeros(D_full.shape[0])
    # set the PTV dose
    target_dose[cfg.OBJ[cfg.PTV_structure]['IDX']] = cfg.PTV_dose
    # set the OAR target dose to a threshold to prevent penalizing small violations
    target_dose[OAR_indices] = cfg.OAR_threshold
    # set the BDY target dose to a threshold to prevent penalizing small violations
    target_dose[BDY_indices] = cfg.BODY_threshold


    # set D and overwrite target_dose to only consider BODY, OAR, and PTV voxels,
    # i.e., skip all other voxels outside the actual BODY
    D = sp.vstack((D_full[BDY_indices],
                          D_full[OAR_indices],
                          D_full[PTV_indices]))
    target_dose = np.hstack((target_dose[BDY_indices],
                             target_dose[OAR_indices],
                             target_dose[PTV_indices]))


    # D_BDY = D[:n_BDY]
    # D_OAR = D[n_BDY:(n_BDY+n_OAR)]
    # D_PTV = D[(n_BDY+n_OAR):]
    # target_dose_PTV = target_dose[(n_BDY+n_OAR):]


    def identity(z):
        return z

    modifier = identity
    if loss == 'squared':
        modifier = np.square


    # create a new OBJ object to use for the DVH plots
    # OBJ_DVH = {}
    # # inlude the PTV
    # OBJ_DVH[PTV_structure] = dict(OBJ[PTV_structure])
    # for structure in OAR_structures+[BODY_structure]:
    #     # include the relevant structures
    #     OBJ_DVH[structure] = dict(OBJ[structure])
    #     # subtract PTV voxels
    #     OBJ_DVH[structure]['IDX'] = np.setdiff1d(OBJ[structure]['IDX'], OBJ[PTV_structure]['IDX'])

################################################################################

    #
    # Compute Scores
    #

    # set the weights for the probing/surrogate function
    weights = np.zeros(D.shape[0])
    weights[:n_BDY] =              1.0/n_BDY
    weights[n_BDY:(n_BDY+n_OAR)] = 1.0/n_OAR
    weights[(n_BDY+n_OAR):] =      1.0/n_PTV

    # run projected gradient descent
    eta = -1.0 # standard from Seb's config, may be set differently elsewhere
    steps = 20
    eta = 4.48911747
    steps = 100
    res = compute_scores(D, target_dose, weights, eta, steps)

    x_hist, loss_hist, score_residual_hist, score_gradnorm_hist, time_PGD = res


    for r in range(repetitions):
        # set the weights
        weights = np.zeros(D.shape[0])
        weights[:n_BDY] =              1.0/(n_BDY*modifier(cfg.BODY_threshold))
        weights[n_BDY:(n_BDY+n_OAR)] = 1.0/(n_OAR*modifier(cfg.OAR_threshold))
        weights[(n_BDY+n_OAR):] =      1.0/(n_PTV*modifier(cfg.PTV_dose))

        # set the scores
        # default is uniform
        scores = None
        if score_method == 'uniform':
            scores = np.ones(D.shape[0])
        elif score_method == 'gradnorm':
            scores = score_gradnorm_hist[-1]
        elif score_method == 'residual':
            scores = score_residual_hist[-1]

        # get the indices and new weights
        # use r as seed
        sample_idx, new_weights = compute_subset(D, weights, m, r, scores)

        print(f'm={m} sampled {len(sample_idx)} points ({len(sample_idx)/m*100}%) of which {len(np.unique(sample_idx))} aka {len(np.unique(sample_idx))/len(sample_idx)*100}% are unique')
        sample_idx, sample_idx_counts = np.unique(sample_idx, return_counts=True)
        new_weights[sample_idx] = np.multiply(sample_idx_counts, new_weights[sample_idx])
        print(f'\t m={m} sampled {len(sample_idx)} points ({len(sample_idx)/m*100}%) of which {len(np.unique(sample_idx))} aka {len(np.unique(sample_idx))/len(sample_idx)*100}% are unique')

        sample_idx_BDY = sample_idx[sample_idx < n_BDY]
        sample_idx_OAR = sample_idx[np.logical_and(sample_idx >= n_BDY, sample_idx < (n_BDY+n_OAR))]
        sample_idx_PTV = sample_idx[sample_idx >= (n_BDY+n_OAR)]

        sample_n_BDY = len(sample_idx_BDY)
        sample_n_OAR = len(sample_idx_OAR)
        sample_n_PTV = len(sample_idx_PTV)

        if sample_n_BDY < 5:
            print(f'WARNING: only {sample_n_BDY} BODY voxels have been chosen, increase m which is currently m={m}')
            print('SKIP run')
            continue

        if sample_n_OAR < 5:
            print(f'WARNING: only {sample_n_OAR} OAR voxels have been chosen, increase m which is currently m={m}')
            print('SKIP run')
            continue

        if sample_n_PTV < 5:
            print(f'WARNING: only {sample_n_PTV} PTV voxels have been chosen, increase m which is currently m={m}')
            print('SKIP run')
            continue

        sample_D_BDY = D[sample_idx_BDY]
        sample_D_OAR = D[sample_idx_OAR]
        sample_D_PTV = D[sample_idx_PTV]

        sp.save_npz(f'CORT/binaries/{case}_sample_{m}_D_BDY.npz', sample_D_BDY)
        sp.save_npz(f'CORT/binaries/{case}_sample_{m}_D_OAR.npz', sample_D_OAR)
        sp.save_npz(f'CORT/binaries/{case}_sample_{m}_D_PTV.npz', sample_D_PTV)