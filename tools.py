from scipy import linalg
import scipy.sparse as sp
from abc import ABC
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
    
class LU2(ABC):
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
    
    def get_update(self,pos, insert):
        permuted_pos = np.where(self.PT[pos,:] == 1)[0][0]
        P = np.eye(self.size)
        ind = np.concatenate((np.arange(permuted_pos), np.arange(permuted_pos+1,self.size),[permuted_pos]))
        P = P[ind]
        PT = P.T
        Linv_X = self.U.copy()
        Linv_X[:,permuted_pos] = self.Z@linalg.solve_triangular(self.L,self.P1T@insert,lower=True)
        Linv_X = P@Linv_X@PT
        spike = Linv_X[-1].copy()
        ls = len(spike)-1
        E = np.eye(ls+1)
        for i in range(ls):
            if spike[i] != 0:
                E[ls,i] = -spike[i]/Linv_X[i,i]
                spike += E[ls,i]*Linv_X[i,:]
        # print(f"New U: \n{np.round(E@Linv_X,3)}")
        Z = E@P@self.Z
        U = E@Linv_X
        return PT, E, Z, U

class LU_from_B(LU2):
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

class LU_from_update(LU2):
    def __init__(self, ref, U, PT, Z, E):
        self.size = ref.size
        self.P1T = ref.P1T
        self.L = ref.L
        self.U = U
        self.PT = PT
        self.Z = Z
        self.E = E
        
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
    def __init__(self, B_indb):
        self.B_indb = B_indb
        self.pivots = deque()

class MOLPP:
    def __init__(self,A,b,C, n_vars, n_basic):
        self.A = A
        self.C = C
        self.b = b
        self.n_vars = n_vars
        self.n_basic = n_basic

class Vertex():
    def __init__(self):
        pass
    
    def new(self, B_indb, AbC):
        self.AbC = AbC
        self.B_indb = B_indb
        self.B_ind = bin2ind(self.B_indb, self.AbC.n_vars)
        self.N_ind = bin2ind(~B_indb, self.AbC.n_vars)
        B = self.AbC.A[:,self.B_ind]
        self.N = self.AbC.A[:,self.N_ind]
        self.CB = self.AbC.C[:,self.B_ind]
        self.CN = self.AbC.C[:,self.N_ind]
        self.B_inv = LU_from_B(B)
        self.BinvN = self._BinvN(self.N)
        self.CN_eff = self._CN_eff()
        self.Binvb = self._Binvb()

    def pivot(self, parent, piv, B_indb):
        #self.parent = parent
        self.AbC = parent.AbC
        self.B_ind = parent.B_ind.copy()
        self.N_ind = parent.N_ind.copy()
        B_in = parent.N_ind[piv[1]]
        B_out = parent.B_ind[piv[0]]
        self.B_ind[piv[0]] = B_in
        self.N_ind[piv[1]] = B_out
        self.B_indb = B_indb
        #self.B_indb = (parent.B_indb & ~(1 << B_out)) | (1 << B_in)
        #self.N_indb = ~self.B_indb
        self.CB = parent.CB.copy()
        self.CN = parent.CN.copy()
        self.CB[:,piv[0]] = self.AbC.C[:,B_in]
        self.CN[:,piv[1]] = self.AbC.C[:,B_out]

        insert = self.AbC.A[:,parent.N_ind[piv[1]]]
        PT, E, Z, U = parent.B_inv.get_update(piv[0], insert)
        self.B_inv = LU_from_update(parent.B_inv, U, PT, Z, E)

        self.N = parent.N.copy() #self.A[:,self.N_ind]
        self.N[:,piv[1]] = self.AbC.A[:,B_out]
        self.BinvN = self._BinvN(self.N)
        self.CN_eff = self._CN_eff()
        self.Binvb = self._Binvb()

    def pivot2(self, parent, piv, B_indb):
        self.AbC = parent.AbC
        self.B_indb = B_indb
        self.B_ind = bin2ind(self.B_indb, self.AbC.n_vars)
        self.N_ind = bin2ind(~B_indb, self.AbC.n_vars)
        #B = self.AbC.A[:,self.B_ind]
        self.N = self.AbC.A[:,self.N_ind]
        self.CB = self.AbC.C[:,self.B_ind]
        self.CN = self.AbC.C[:,self.N_ind]

        insert = self.AbC.A[:,parent.N_ind[piv[1]]]
        PT, E, Z, U = parent.B_inv.get_update(piv[0], insert)
        self.B_inv = LU_from_update(parent.B_inv, U, PT, Z, E)
        #self.B_inv = LU_from_B(B)
        self.BinvN = self._BinvN(self.N)
        self.CN_eff = self._CN_eff()
        self.Binvb = self._Binvb()


    def _BinvN(self, N):
        return self.B_inv.solve(N)
    
    def _CN_eff(self):
        return self.CN-self.CB@self.BinvN
    
    def _Binvb(self):
        return self.B_inv.solve(self.AbC.b)

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

def set_masks(cfg):
    for key in cfg.obj.keys():
        mask = np.zeros(cfg.n_voxels)
        mask[cfg.OBJ[key]['IDX']] = 1.0
        cfg.OBJ[key]['MASK'] = mask.reshape(cfg.dim)

def plot_results(case = 'Liver', results_file=None, prefix="",slice_=50):
    cfg = utils.get_config(case)

    if not results_file:
        if case=='Liver':
            res = np.load('result_liver_BDY_downsample_10000_OAR_downsample_1000_PTV_downsample_100.npz')
            slice_ = 42
        if case=='Prostate':
            res = np.load('result_prostate_BDY_downsample_3000_OAR_downsample_300_PTV_downsample_30.npz')
            slice_ = 55
    else:
        res = np.load(results_file)

    D_full = CORT.load_D_full(case)
    solutions = res['array_data'][:,:cfg.n_vars]
    if prefix != "":
        prefix = prefix + "_"

    keys = [cfg.BODY_structure] + cfg.OAR_structures + [cfg.PTV_structure]
    D_patient = D_full[cfg.OBJ[cfg.BODY_structure]['IDX']]

    set_masks(cfg)

    i=0
    for s in solutions:
        dose = np.zeros(cfg.n_voxels)
        dose[cfg.OBJ[cfg.BODY_structure]['IDX']] = D_patient@s
        dose = dose.reshape(cfg.dim)

        #for slice_ in np.arange(0, dim[0], 5):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        #x1, x2 = 10, 160
        #y1, y2 = 30, 130
        doseplt = ax.imshow(dose[slice_,:,:].T, cmap='hot', alpha=1)
        for key in keys:
            #con = ax.contour(OBJ[key]['MASK'][slice_,x1:x2,y1:y2].T, levels=[0.5], colors=OBJ[key]['COLOR'])
            con = ax.contour(cfg.OBJ[key]['MASK'][slice_,:,:].T, levels=[0.5], colors=cfg.OBJ[key]['COLOR'])
            # dirty hack to check whether the contour is empty
            if con.levels[0] > 0.0:
                # add label for legend
                ax.plot([], [], c=cfg.OBJ[key]['COLOR'], label=key)
        cbar = fig.colorbar(doseplt, ax=ax, label='Radiation Dose (Gy)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'plots/{case}/result_{prefix}{case}_{i}_{slice_}.png', dpi=300, transparent=True, bbox_inches='tight')
        i+=1

def plot_first_result(case = 'Liver', results_file=None, prefix=""):
    cfg = utils.get_config(case)

    if not results_file:
        if case=='Liver':
            res = np.load('results/result_liver_BDY_downsample_10000_OAR_downsample_1000_PTV_downsample_100.npz')
            slice_ = 42
        if case=='Prostate':
            res = np.load('results/result_prostate_BDY_downsample_3000_OAR_downsample_300_PTV_downsample_30.npz')
            slice_ = 55
    else:
        res = np.load(results_file)

    #solvec = res['array_data'][:,:389][0]
    D_full = CORT.load_D_full(case)
    solutions = res['array_data'][:,:cfg.n_vars]
    if prefix != "":
        prefix = prefix + "_"

    keys = [cfg.BODY_structure] + cfg.OAR_structures + [cfg.PTV_structure]
    D_patient = D_full[cfg.OBJ[cfg.BODY_structure]['IDX']]

    set_masks(cfg)

    s = solutions[0]
    dose = np.zeros(cfg.n_voxels)
    dose[cfg.OBJ[cfg.BODY_structure]['IDX']] = D_patient@s
    dose = dose.reshape(cfg.dim)
    for slice_ in np.arange(0, cfg.dim[0], 5):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        #x1, x2 = 10, 160
        #y1, y2 = 30, 130
        doseplt = ax.imshow(dose[slice_,:,:].T, cmap='hot', alpha=1)
        for key in keys:
            #con = ax.contour(OBJ[key]['MASK'][slice_,x1:x2,y1:y2].T, levels=[0.5], colors=OBJ[key]['COLOR'])
            con = ax.contour(cfg.OBJ[key]['MASK'][slice_,:,:].T, levels=[0.5], colors=cfg.OBJ[key]['COLOR'])
            # dirty hack to check whether the contour is empty
            if con.levels[0] > 0.0:
                # add label for legend
                ax.plot([], [], c=cfg.OBJ[key]['COLOR'], label=key)
        cbar = fig.colorbar(doseplt, ax=ax, label='Radiation Dose (Gy)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'plots/{case}/all_slices/result_{prefix}{case}_{slice_}.png', dpi=300, transparent=True, bbox_inches='tight')

def plot_slices(case):
    """
    Plots slices with contours of structures only
    """
    cfg = utils.get_config(case)

    CORT.load_indices(cfg)

    keys = [cfg.BODY_structure] + cfg.OAR_structures + [cfg.PTV_structure]

    set_masks(cfg)

    for slice_ in np.arange(0, cfg.dim[0], 5):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        #x1, x2 = 10, 160
        #y1, y2 = 30, 130
        for key in keys:
            #con = ax.contour(OBJ[key]['MASK'][slice_,x1:x2,y1:y2].T, levels=[0.5], colors=OBJ[key]['COLOR'])
            con = ax.contour(cfg.OBJ[key]['MASK'][slice_,:,:].T, levels=[0.5], colors=cfg.OBJ[key]['COLOR'])
            # dirty hack to check whether the contour is empty
            if con.levels[0] > 0.0:
                # add label for legend
                ax.plot([], [], c=cfg.OBJ[key]['COLOR'], label=key)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'plots/slices/slice_{case}_{slice_}.png', dpi=300, transparent=True, bbox_inches='tight')

def get_mean_doses(case='Prostate', results_file=None):
    cfg = utils.get_config(case)
    
    #BDY_indices, OAR_indices, PTV_indices, n_BDY, n_OAR, n_PTV = utils.get_diff_indices(cfg)

    D_BDY, D_OAR, D_PTV = CORT.load_D_XYZ(case)
    BDY_indices, OAR_indices, PTV_indices, n_BDY, n_OAR, n_PTV = utils.get_diff_indices(cfg,True)
    D_full = CORT.load_D_full(case)
    D_BDY = D_full[BDY_indices]
    D_OAR = D_full[OAR_indices]
    D_PTV = D_full[PTV_indices]

    if not results_file:
        if case=='Liver':
            res = np.load('result_liver_BDY_downsample_10000_OAR_downsample_1000_PTV_downsample_100.npz')
        if case=='Prostate':
            res = np.load('result_prostate_BDY_downsample_3000_OAR_downsample_300_PTV_downsample_30.npz')
    else:
        res = np.load(results_file)
    length_t = D_BDY.shape[1]
    solutions = res['array_data'][:,:length_t]
    num_solutions = len(solutions)

    mean_doses = {'BDY':[0]*num_solutions,
                  'OAR':[0]*num_solutions,
                  'PTV':[0]*num_solutions}
    print(np.mean(D_full@solutions[0]))
    for i in range(num_solutions):
        mean_doses['BDY'][i] = np.mean(D_BDY@solutions[i])
        mean_doses['OAR'][i] = np.mean(D_OAR@solutions[i])
        mean_doses['PTV'][i] = np.mean(D_PTV@solutions[i])
    
    # for i in range(num_solutions):
    #     mean_doses['BDY'][i] /= n_BDY
    #     mean_doses['OAR'][i] /= n_OAR
    #     mean_doses['PTV'][i] /= n_PTV

    return mean_doses