import numpy as np
from LUsolve import LUsolve

def solve(C,A,b,B,Basic,nonBasic,x0):
    c = C.sum(axis=0)
    Basic = np.concatenate((Basic, np.arange(len(c), len(c) + C.shape[0])))
    b = np.concatenate((b, C @ x0))
    c = np.concatenate((c,np.zeros(C.shape[0])))
    A = np.hstack((A, np.zeros((A.shape[0], C.shape[0]))))
    C = np.hstack(C, np.eye(C.shape[0]))
    A = np.vstack(A,C)

    # b_bar = np.zeros(B.shape(0))
    # cn = np.zeros(len(c)-len(b))

    B = A[:,Basic]
    
    maxIterations = 50
    binv = LUsolve(B)
    for i in range(maxIterations):
        
        b_bar = binv.solve(b)
        cn = c[nonBasic]-c[Basic]@binv.solve(A[:,nonBasic])

        if np.all(cn <= 1e-5):
            return Basic
        
        s = np.argmax(cn)
        a_s_bar = binv.solve(A[:,s])
        valid = a_s_bar> 0
        values = np.where(valid, b_bar / a_s_bar, np.inf)
        min_index = np.argmin(values)
        if values[min_index] == np.inf:
            print("Problem is unbounded!")
            return []

        tmp = nonBasic[s]
        nonBasic[s] = Basic[min_index]
        Basic[min_index] = tmp

        B = A[:,Basic]
        binv = LUsolve(B)



