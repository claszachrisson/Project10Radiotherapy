import numpy as np
from LUsolve import LUsolve

def solve(C,A,b,B,Basic,nonBasic,x0):
    c = C.sum(axis=0)
    Basic_s = np.concatenate((Basic, np.arange(len(c), len(c) + C.shape[0])))
    nonBasic_s = nonBasic.copy()
    b_s = np.concatenate((b, C @ x0))
    c = np.concatenate((c,np.zeros(C.shape[0])))
    A_s = np.hstack((A, np.zeros((A.shape[0], C.shape[0]))))
    C_s = np.hstack((C, np.eye(C.shape[0])))
    A_s = np.vstack((A_s,C_s))

    # b_bar = np.zeros(B.shape(0))
    # cn = np.zeros(len(c)-len(b))

    # print("c",c)
    # print("b",b_s)
    # print("A",A_s)
    # print("C",C_s)


    B_s = A_s[:,Basic_s]
    
    maxIterations = 50
    binv = LUsolve(B_s)
    for i in range(maxIterations):
        
        try:
            b_bar = binv.solve(b_s)
        except ValueError:
            return []
            # if np.any(np.isnan(b_bar)):
            #     return []
        cn = c[nonBasic_s]-c[Basic_s]@binv.solve(A_s[:,nonBasic_s])

        if np.any(np.isnan(cn)):
            return []
        # print(cn)
        if np.all(np.abs(cn) <= 1e-5):
            if(np.all(np.abs(binv.solve(b)[Basic_s])>1e-5)):
                print("BINV",binv.solve(b)[Basic_s])
                # print(cn)
                return Basic_s
        
        s = np.argmax(cn)
        a_s_bar = binv.solve(A_s[:,s])
        valid = a_s_bar> 1e-6
        values = np.where(valid, b_bar / a_s_bar, np.inf)
        min_index = np.argmin(values)
        if values[min_index] == np.inf:
            print("Problem is unbounded!")
            return []

        tmp = nonBasic_s[s]
        nonBasic_s[s] = Basic_s[min_index]
        Basic_s[min_index] = tmp

        B_s = A_s[:,Basic_s]
        binv = LUsolve(B_s)
    return []

A = np.array([[1, 1, 2], [1, 2, 1], [2, 1, 1]])
A = np.hstack((A, np.eye(A.shape[0])))  # Add identity matrix to A
b = np.array([12, 12, 12])
C = np.array([[6, 4, 5]])
C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))  # Add zero columns to C

Basic = [3,4,5]
nonBasic = [0,1,2]
x0 = [0,0,0,12,12,12]
B = A[:,Basic]

res = solve(C,A,b,B,Basic,nonBasic,x0)
print(res)