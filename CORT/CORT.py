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

def load_data(data_path, OBJ, case):
    # angles is a list of lists [[gantry_angle, couch_angle],...]

    # load the structure indices
    for key in OBJ.keys():
        OBJ[key]['IDX'] = load_structure(f'CORT/VOILISTS/{key}_VOILIST.mat')

    D_full = sp.load_npz(f'CORT/binaries/{case}_D_full.npz')

    return D_full


