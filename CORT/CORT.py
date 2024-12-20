import scipy.sparse as sp

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

def load_indices(OBJ, case):
    # load the structure indices
    for key in OBJ.keys():
        OBJ[key]['IDX'] = load_structure(f'CORT/VOILISTS/{case}/{key}_VOILIST.mat')

def load_D_full(case):
    return sp.load_npz(f'CORT/binaries/{case}_D_full.npz')