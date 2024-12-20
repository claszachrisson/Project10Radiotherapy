from scipy import linalg
import numpy as np
import os, sys
from collections import deque
import matplotlib.pyplot as plt
import CORT.utils as utils
import CORT.CORT as CORT

class LU:
    def __init__(self, B):
        self.B = np.array(B)
        self.size = self.B.shape[0]
        self.LU, self.piv = linalg.lu_factor(self.B)

    def solve(self,b):
        b = np.array(b)
        if(b.ndim > 1):
            x = np.zeros([self.size, b.shape[1]])
            for i in range(b.shape[1]):
                x[:,i] = linalg.lu_solve((self.LU,self.piv),b[:,i])
            return x
        else:
            return linalg.lu_solve((self.LU,self.piv),b)
    
class LU2:
    def __init__(self, B):
        B = np.array(B)
        if(B.shape[0] != B.shape[1]):
            raise ValueError("Matrix not square.")
        self.size = B.shape[0]
        self.P1T, self.L, self.U = linalg.lu(B)
        self.P1T = self.P1T.T
        self.PT = np.eye(self.size)
        self.E = np.eye(self.size)
        self.Z = np.eye(self.size)
        self.P1_backup, self.L_backup, self.U_backup = linalg.lu(B)
        self.PT_backup = np.eye(self.size)
        self.E_backup = np.eye(self.size)
        self.Z_backup = np.eye(self.size)
    
    def solve(self,b):
        y = linalg.solve_triangular(self.L,self.P1T@b,lower=True)
        x = linalg.solve_triangular(self.U,self.Z@y)
        return self.PT@x
    
    def solveMat(self,N):
        c = N.shape[1]
        x = np.zeros([self.size, c])
        for i in range(c):
            x[:,i] = self.solve(N[:,i])
        return x
    
    def update(self,pos,insert):
        permuted_pos = np.where(self.PT[pos,:] == 1)[0][0]
        #print(pos, permuted_pos)
        P = np.eye(self.size)
        ind = np.concatenate((np.arange(permuted_pos), np.arange(permuted_pos+1,self.size),[permuted_pos]))
        P = P[ind]
        PT = P.T
        Linv_X = self.U.copy()
        Linv_X[:,permuted_pos] = self.Z@linalg.solve_triangular(self.L,self.P1T@insert,lower=True)
        # print(f"Inserted Linv_X: \n{np.round(Linv_X,3)}")
        # print(f"Permutation matrix: \n{np.round(P,3)}")
        self.PT = self.PT@PT
        Linv_X = P@Linv_X@PT
        # print(f"Permuted Linv_X: \n{np.round(Linv_X,3)}")
        spike = Linv_X[-1].copy()
        ls = len(spike)-1
        E = np.eye(ls+1)
        for i in range(ls):
            if spike[i] != 0:
                E[ls,i] = -spike[i]/Linv_X[i,i]
                spike += E[ls,i]*Linv_X[i,:]
        # print(f"New U: \n{np.round(E@Linv_X,3)}")
        self.Z = E@P@self.Z
        self.U = E@Linv_X

    def backup(self):
        self.P1_backup = self.P1_backup.copy()
        self.L_backup = self.L.copy()
        self.U_backup = self.U.copy()
        self.PT_backup = self.PT.copy()
        self.E_backup = self.E.copy()
        self.Z_backup = self.Z.copy()

    def revert2backup(self):
        self.P1 = self.P1_backup.copy()
        self.L = self.L_backup.copy()
        self.U = self.U_backup.copy()
        self.PT = self.PT_backup.copy()
        self.E = self.E_backup.copy()
        self.Z = self.Z_backup.copy()

        
class Matrices():
    def __init__(self,A,b,C, B_indb, n_vars):
        self.A = A
        self.C = C
        self.b = b
        self.n_vars = n_vars
        self.update(B_indb)

    def update(self,B_indb):
        self.B_indb = B_indb
        self.N_indb = ~B_indb
        self.B_ind = bin2ind(self.B_indb, self.n_vars)
        self.N_ind = bin2ind(self.N_indb, self.n_vars)
        self.B = self.A[:,self.B_ind]
        self.CB = self.C[:,self.B_ind]
        self.N = self.A[:,self.N_ind]
        self.CN = self.C[:,self.N_ind]
        self.B_inv = LU2(self.B)
        self.BinvN = self._BinvN()
        self.CN_eff = self._CN_eff()
        self.Binvb = self._Binvb()

    def update2(self,B_indb, pivot):
        self.B_indb = B_indb
        self.N_indb = ~B_indb
        self.B_ind = bin2ind(self.B_indb, self.n_vars)
        self.N_ind = bin2ind(self.N_indb, self.n_vars)
        self.B = self.A[:,self.B_ind]
        self.CB = self.C[:,self.B_ind]
        self.N = self.A[:,self.N_ind]
        self.CN = self.C[:,self.N_ind]
        #print(f'Inserting index {pivot[2]} into index {pivot[0]}, B={self.B_ind}')
        self.B_inv.update(pivot[0], self.A[:,pivot[2]])
        self.BinvN = self._BinvN()
        self.CN_eff = self._CN_eff()
        self.Binvb = self._Binvb()
    
    # def pivot(self, swap):
    #     self.B_indb = (self.B_indb & ~(1 << swap[0])) | (1 << swap[1])
    #     self.N_indb = ~self.B_indb
    #     print(f"attempting swap {swap} from {self.B_ind}")
    #     B_out_index = self.B_ind.index(swap[0])
    #     N_out_index = self.N_ind.index(swap[1])
    #     self.B_ind[B_out_index] = swap[1]
    #     self.N_ind[N_out_index] = swap[0]
    #     self.B[:,B_out_index] = self.A[:,swap[1]]
    #     self.N[:,N_out_index] = self.A[:,swap[0]]
    #     self.CB[:,B_out_index] = self.C[:,swap[1]]
    #     self.CN[:,N_out_index] = self.C[:,swap[0]]
    #     self.B_inv = LU(self.B)
    #     self.BinvN[:,N_out_index] = self.B_inv.solve(self.C[:,swap[0]])
    #     self.CN_eff[:,N_out_index] = self.C[:,swap[0]]-self.CB@self.BinvN[:,N_out_index]
    #     self.Binvb = self._Binvb()

    def _BinvN(self):
        return self.B_inv.solve(self.N)
    
    def _CN_eff(self):
        return self.CN-self.CB@self.BinvN
    
    def _Binvb(self):
        return self.B_inv.solve(self.b)

class Basis:
    def __init__(self, B_indb):#, lu_obj):
        self.B_indb = B_indb
        self.pivots = deque()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def bin2ind(bin, n_var):
    return [i for i in range(n_var) if (bin & (1 << i))]

def ind2bin(ind):
    bin = 0
    for i in ind:
        bin = bin | (1 << int(i))
    return bin




def get_dim(case):
    if case == 'Prostate':
        dim = np.array([184,184,90])
    if case == 'Liver':
        dim = np.array([217,217,168])
    if case == 'HeadAndNeck':
        dim = np.array([160,160,67])
    return dim


def get_mask(obj,nVoxels,dim):
    mask = np.zeros(nVoxels)
    mask[obj] = 1.0
    return mask.reshape(dim)



def plot_results(case = 'Liver'):
    cfg = utils.get_config(case)
    data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BDY_threshold, OAR_structures, OAR_threshold = cfg

    if case=='Liver':
        res = np.load('result_liver_BDY_downsample_10000_OAR_downsample_1000_PTV_downsample_100.npz')
        slice_ = 42
    if case=='Prostate':
        res = np.load('result_prostate_BDY_downsample_3000_OAR_downsample_300_PTV_downsample_30.npz')
        slice_ = 55
    #solvec = res['array_data'][:,:389][0]
    D_full = CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))
    length_t = D_full.shape[1]
    solutions = res['array_data'][:,:length_t]

    keys = [BODY_structure] + OAR_structures + [PTV_structure]
    dim = get_dim(case)
    dim = np.roll(dim, 1)
    nVoxels = np.prod(dim)
    D_patient = D_full[OBJ[BODY_structure]['IDX']]

    for key in keys:
        OBJ[key]['MASK'] = get_mask(OBJ[key]['IDX'], nVoxels,dim)
    i=0
    for s in solutions:
        dose = np.zeros(nVoxels)
        dose[OBJ[BODY_structure]['IDX']] = D_patient@s
        dose = dose.reshape(dim)

        #for slice_ in np.arange(0, dim[0], 5):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        #x1, x2 = 10, 160
        #y1, y2 = 30, 130
        doseplt = ax.imshow(dose[slice_,:,:].T, cmap='hot', alpha=1)
        for key in keys:
            #con = ax.contour(OBJ[key]['MASK'][slice_,x1:x2,y1:y2].T, levels=[0.5], colors=OBJ[key]['COLOR'])
            con = ax.contour(OBJ[key]['MASK'][slice_,:,:].T, levels=[0.5], colors=OBJ[key]['COLOR'])
            # dirty hack to check whether the contour is empty
            if con.levels[0] > 0.0:
                # add label for legend
                ax.plot([], [], c=OBJ[key]['COLOR'], label=key)
        cbar = fig.colorbar(doseplt, ax=ax, label='Radiation Dose (Gy)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'plots/{case}/result_{case}_{i}_{slice_}.png', dpi=300, transparent=True, bbox_inches='tight')
        i+=1

def plot_slices(case):
    """
    Plots slices with contours of structures only
    """
    cfg = utils.get_config(case)
    data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BODY_threshold, OAR_structures, OAR_threshold = cfg

    CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))

    dim = get_dim(case)

    dim = np.roll(dim, 1)
    nVoxels = np.prod(dim)
    keys = [BODY_structure] + OAR_structures + [PTV_structure]

    for key in keys:
        OBJ[key]['MASK'] = get_mask(OBJ[key]['IDX'], nVoxels,dim)


    for slice_ in np.arange(0, dim[0], 5):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        #x1, x2 = 10, 160
        #y1, y2 = 30, 130
        for key in keys:
            #con = ax.contour(OBJ[key]['MASK'][slice_,x1:x2,y1:y2].T, levels=[0.5], colors=OBJ[key]['COLOR'])
            con = ax.contour(OBJ[key]['MASK'][slice_,:,:].T, levels=[0.5], colors=OBJ[key]['COLOR'])
            # dirty hack to check whether the contour is empty
            if con.levels[0] > 0.0:
                # add label for legend
                ax.plot([], [], c=OBJ[key]['COLOR'], label=key)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'plots/slices/slice_{case}_{slice_}.png', dpi=300, transparent=True, bbox_inches='tight')