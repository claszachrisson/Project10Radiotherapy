import scipy as sp
import numpy as np
from os.path import exists

def load_structure(file):
    if exists(file):
        MAT = sp.io.loadmat(file)
        # subtract 1 due to indexing mismatch
        # MATLAB starts with 1 and Python with 0
        return MAT['v'].reshape(-1) - 1
    else:
        print(f'ERROR: file {file} not found')
        return None

def load_indices(cfg=None, intersect=False):
    # load the structure indices
    for key in cfg.OBJ.keys():
        cfg.OBJ[key]['IDX'] = load_structure(f'CORT/VOILISTS/{cfg.case}/{key}_VOILIST.mat')
    if not intersect:
        # set the indices for body (BDY), OAR, and PTV
        BDY_indices = cfg.OBJ[cfg.BODY_structure]['IDX']
        PTV_indices = cfg.OBJ[cfg.PTV_structure]['IDX']
        OAR_indices = np.unique(np.hstack([cfg.OBJ[cfg.OAR_structure]['IDX'] for OAR_structure in cfg.OAR_structures]))
        # fix the indices
        OAR_indices = np.setdiff1d(OAR_indices, PTV_indices)
        BDY_indices = np.setdiff1d(BDY_indices, np.union1d(PTV_indices, OAR_indices))

        assert len(np.intersect1d(BDY_indices, PTV_indices)) == 0
        assert len(np.intersect1d(OAR_indices, PTV_indices)) == 0
        assert len(np.intersect1d(OAR_indices, BDY_indices)) == 0
        
        cfg.OBJ[cfg.BODY_structure]['IDX'] = BDY_indices

def load_D_full(case):
    return sp.sparse.load_npz(f'CORT/binaries/{case}_D_full.npz')

def load_D_XYZ(case, lengths=False):
    D_BDY = sp.sparse.load_npz(f'CORT/binaries/{case}_D_BDY.npz')
    D_OAR = sp.sparse.load_npz(f'CORT/binaries/{case}_D_OAR.npz')
    D_PTV = sp.sparse.load_npz(f'CORT/binaries/{case}_D_PTV.npz')
    if lengths:
        return D_BDY, D_OAR, D_PTV, len(D_BDY), len(D_OAR), len(D_PTV)
    return D_BDY, D_OAR, D_PTV