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