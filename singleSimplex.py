import numpy as np
from tools import LU

def solveLin(C,A,b,Basic,nonBasic,x0):
    """
    Attempts to solve the linear programming problem given by step 6 in the algorithm.

    Returns the solution vector x
    """

    c = C.sum(axis=0)
    Basic_s = np.concatenate((Basic, np.arange(len(c), len(c) + C.shape[0])))
    nonBasic_s = nonBasic.copy()
    b_s = np.concatenate((b, C @ x0))
    c = np.concatenate((c,np.zeros(C.shape[0])))
    A_s = np.hstack((A, np.zeros((A.shape[0], C.shape[0]))))
    C_s = np.hstack((C, -1*np.eye(C.shape[0])))
    A_s = np.vstack((A_s,C_s))

    B_s = A_s[:,Basic_s]
    
    maxIterations = 20
    for _ in range(maxIterations):
        B_s = A_s[:,Basic_s]

        if np.linalg.cond(B_s) > 1e12:
            x = np.zeros(A.shape[1])
            return x
        
        binv = LU(B_s)
        
        b_bar = binv.solve(b_s)
        cn = c[nonBasic_s]-np.transpose(binv.solve(A_s[:,nonBasic_s]))@c[Basic_s]

        if np.all(cn <= 1e-5):
            x = np.zeros(A_s.shape[1])
            x[Basic_s] = b_bar
            return x[:A.shape[1]]
        
        s = np.argmax(cn)
        a_s_bar = binv.solve(A_s[:,s])
        valid = a_s_bar> 1e-6
        values = np.where(valid, b_bar / a_s_bar, np.inf)
        min_index = np.argmin(values)
        if values[min_index] == np.inf:
            print("Problem is unbounded!")
            x = np.zeros(A.shape[1])
            return x

        tmp = nonBasic_s[s]
        nonBasic_s[s] = Basic_s[min_index]
        Basic_s[min_index] = tmp

    x = np.zeros(A.shape[1])
    return x