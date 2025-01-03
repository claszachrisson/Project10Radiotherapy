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

def load_indices(cfg):
    # load the structure indices
    for key in cfg.OBJ.keys():
        cfg.OBJ[key]['IDX'] = load_structure(f'{cfg.data_path}/VOILISTS/{cfg.case}/{key}_VOILIST.mat')

def load_D_full(cfg):
    return sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.case}_D_full.npz')

def load_D_XYZ(cfg, lengths=False):
    D_BDY = sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.filenames}_D_BDY.npz')
    D_OAR = sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.filenames}_D_OAR.npz')
    D_PTV = sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.filenames}_D_PTV.npz')
    if lengths:
        return D_BDY, D_OAR, D_PTV, len(D_BDY), len(D_OAR), len(D_PTV)
    return D_BDY, D_OAR, D_PTV