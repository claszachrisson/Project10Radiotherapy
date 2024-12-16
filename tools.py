from scipy import linalg
import numpy as np
import os, sys

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