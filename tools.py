from scipy import linalg
import numpy as np
import os, sys
from collections import deque

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
        self.P, self.L, self.U = linalg.lu(B)
        self.P2 = np.eye(self.size)
        self.E = self.P2
    
    def solve(self,b):
        y = linalg.solve_triangular(self.L,self.P.T@b,lower=True)
        x = linalg.solve_triangular(self.U,self.E@self.P2@y)
        return self.P2.T@x
    
    def update(self,pos,insert):
        ind = np.concatenate((np.arange(pos), np.arange(pos+1,self.size),[pos]))
        self.P2 = self.P2[ind]
        L_Bn = self.U.copy()
        L_Bn[:,pos] = linalg.solve_triangular(self.L,self.P.T@insert,lower=True)
        L_Bn = self.P2@L_Bn@self.P2.T
        spike = L_Bn[-1,:].copy()
        ls = len(spike)-1
        self.E = np.eye(ls+1)
        for i in range(ls):
            if spike[i] != 0:
                self.E[ls,i] = -spike[i]/L_Bn[i,i]
                spike += self.E[ls,i]*L_Bn[i,:]
        self.U = self.E@L_Bn
        
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
        self.B_inv = LU(self.B)
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