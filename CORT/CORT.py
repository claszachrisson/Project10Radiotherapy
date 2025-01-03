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

def load_D_full(cfg): # from npz
    return sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.case}_D_full.npz')

def get_D_full(cfg): # from .mat
    # load the dose influence matrix per gantry angle and concatenate them
    D = []
    for gantry_angle, couch_angle in list(zip(cfg.gantry_angles, cfg.couch_angles)):
        file = f'{cfg.data_path}/Gantry{gantry_angle}_Couch{couch_angle}_D.mat'
        if exists(file):
            beam_D = sp.io.loadmat(file)
            D.append(beam_D['D'])
        else:
            print(f'ERROR: file {file} not found')
    D_full = sp.sparse.hstack(D)

    return D_full

def load_D_XYZ(cfg, lengths=False):
    D_BDY = sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.filenames}_D_BDY.npz')
    D_OAR = sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.filenames}_D_OAR.npz')
    D_PTV = sp.sparse.load_npz(f'{cfg.data_path}/binaries/{cfg.filenames}_D_PTV.npz')
    if lengths:
        return D_BDY, D_OAR, D_PTV, len(D_BDY), len(D_OAR), len(D_PTV)
    return D_BDY, D_OAR, D_PTV