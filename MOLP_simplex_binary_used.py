import numpy as np
# import jax
from scipy.optimize import linprog
import time

from tools import LU, ind2bin, bin2ind, Matrices

def check_efficient(M):

    #Check if any row is positive non-zero
    pos_nonzero = np.any(np.logical_or(M.CN_eff < 0, np.all(M.CN_eff == 0, axis=0)), axis=0) #Check if any column is all negative or if all zero (check correctness)

    #If positive non-zero the solution is not efficient
    if(np.any(pos_nonzero==False)):
        return False
    
    #If any row is negative the solution is efficient
    any_row_negative = np.all(M.CN_eff < 0, axis=1)
    if(np.any(any_row_negative==True)):
        return True
    
    #Calculate the sum of the rows (sum the columns)
    row_sums = M.CN_eff.sum(axis=0)

    #If all are 0 or negative (some small tolerance)
    if(np.all(row_sums<=1e-6)):
        return True
    
    return False

def find_possible_eff_sols(M, used_indices, basic_explore,num_basic):
    As = M.BinvN

    #Cols with mixed components
    cols = [i for i in range(M.CN_eff.shape[1]) if np.any(M.CN_eff[:, i] > 1e-5) and np.any(M.CN_eff[:, i] < 1e-5)]

    # print(len(cols))
    # print(CN_eff)

    #If no mixed columns no possible eff solution in columns
    if(cols==[]):
        return []

    #For pivoting
    b_eff = M.Binvb

    #Only consider mixed component columns
    As=np.array(As[:,cols])

    valid = As > 1e-6  # Boolean array for valid entries
    values = np.where(valid, b_eff[:, None] / As, np.inf)  # Broadcast and compute

    t = np.min(values, axis=0)
    index_outs = np.argmin(values, axis=0)

    index_ins = np.where(t < np.inf, cols, np.inf)
    index_outs = np.where(t < np.inf, index_outs, np.inf)
    mask = np.isfinite(t)
    
    t = t[mask]
    cols = np.array(cols) #Not sure if correct?
    cols = cols[mask]
    tC = t * M.CN_eff[:, cols]
    # print(tC)
    if cols.size==0:
        return []
    ind = []
    if len(t)>1:
        dominance_matrix = np.all(tC[:, :, None] < tC[:, None, :], axis=0)
        np.fill_diagonal(dominance_matrix, False)
        non_dominated = ~np.any(dominance_matrix, axis=0)

        ind = np.where(non_dominated)[0].tolist()
        #print(ind)
    else:
        ind=[0]
    
    if ind == []:
        return []
    
    basic_ind_list = []
    pivot_ins = index_ins[ind].astype(int)
    pivot_outs = index_outs[ind].astype(int)
    for i in range(len(ind)):
        tmp_B_indb = (M.B_indb & ~(1 << int(M.B_ind[pivot_outs[i]]))) | (1 << int(M.N_ind[pivot_ins[i]]))
        if not tmp_B_indb in used_indices and not tmp_B_indb in basic_explore:
            basic_ind_list.append(tmp_B_indb)


    return basic_ind_list


def simplex(A,b,C, std_form = True, Initial_basic = None, num_sol = 100):
    """
    Solves a multi-objective linear programming problem using the Simplex method.

    This function maximizes multiple objective functions subject to a set of linear constraints.
    The multi-objective problem is formulated as:

    max Cx

    s.t Ax=b
        x>=0


    Parameters:
    A (ndarray):  A 2D numpy array representing the constraint matrix (m x n).
    b (ndarray):  A 1D numpy array representing the right-hand side of the constraints (m).
    C (ndarray)   A 2D numpy array with coefficients for each objective function (k x n).
    num_sol (int) An integer with the number of efficient solutions that should be found.

    Returns:
    ndarray: The optimal bases of the solutions.
    ndarray: The efficient solution vectors `x` that maximizes the objectives.
    ndarray: The optimal values of the objective functions at the solutions.

    Notes:
    - This method assumes the problem is feasible.
    - 

    """
    iters = -1
    print("STARTING MOLP SIMPLEX")
    if std_form==False:
        A = np.hstack((A, np.eye(A.shape[0])))
        C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))
    
    num_variables = len(C[0,:])
    num_basic = len(b) #Number of basic variables is the number of constraints
    num_non_basic = num_variables-num_basic #Number of non-basic is the number of columns minus the basic variables

    #Initial feasible solution
    if Initial_basic is not None:
        B_ind = Initial_basic
    else:
        B_ind = list(np.arange(num_non_basic, num_variables))

    B_indb = ind2bin(list(B_ind))
    if(B_indb == 0):
        print("Error when converting B_ind to binary")
        exit()
    #print(f"Converted B_indb from {B_ind}:")
    #print(format(B_indb,f'0{num_variables}b'))

    #Create a list saving for saving the results
    used_indices = []
    eff_ind = []
    solution_vec = []

    #Create the matrices for the initial solution
    M = Matrices(A,b,C,B_indb,num_variables)

    #Loop to find initial efficient solution

    sols=True
    basic_explore = []

    ideal_sol = np.all(M.CN_eff<=0)
    if(ideal_sol==True):
        return True
    
    t = [0,0,0,0,0]
    C_row_sum = -C.sum(axis=0)
    
    while sols:
        iters+=1
        print(f"Iteration {iters}, with solution len {len(eff_ind)}, basic explore: {len(basic_explore)}")
        print(f"Times spent: {[round(tt,3) for tt in t]}")
        tt = time.time()
        eff = check_efficient(M)
        t[0] += time.time()-tt
        if eff:
            print("CHECK EFF")
            solution_vec.append(M.Binvb)
            eff_ind.append(B_ind.copy())
        else:
            tt = time.time()
            x0 = np.zeros(A.shape[1])
            x0[M.B_ind]=M.Binvb

            try:
                tlp = time.time()
                result = linprog(C_row_sum, b_ub = -C@x0, A_ub = -C, A_eq = A, b_eq = b, x0=x0, method="revised simplex")
                t[2] += time.time()-tlp
                sol = result.x
            except Exception as e:
                print(f"An error occurred during the linprog optimization: {e}")
                print("Proceeding!")
                sol=np.zeros(x0.shape)

            tmp_B_ind = -1
            if np.all(sol == x0):
                tmp_B_ind = B_ind
            else:
                tmp_B_ind2 = np.where(np.abs(sol)>1e-6)
                tmp_B_ind2 = list(tmp_B_ind2[0])
                tmp_B_indb = ind2bin(tmp_B_ind2)
                if (len(tmp_B_ind2)==num_basic) and not tmp_B_indb in used_indices:
                    tmp_B_ind = tmp_B_ind2

            if tmp_B_ind != -1:
                print("LINPROG")
                used_indices.append(B_indb)
                B_indb = ind2bin(tmp_B_ind)
                M.update(B_indb)

                # #Add new non-basic list
                # B = A[:,B_ind]
                # N = A[:,N_ind]
                # CB = C[:,B_ind]
                # CN = C[:,N_ind]
                # B_inv = LU(B)

                solution_vec.append(M.Binvb)
                eff_ind.append(B_ind.copy())
            t[1] += time.time() - tt
        
        tt = time.time()
        basic_ind_list = find_possible_eff_sols(M, used_indices,basic_explore,num_basic)
        t[3] += time.time() - tt
        tt = time.time()

        for basic in basic_ind_list:
            basic_explore.append(basic)

        if len(basic_explore)==0:
            sols = False
        else:
            print("Changing basis")
            used_indices.append(B_indb)
            in_used = True
            while in_used:
                B_indb = basic_explore[0]
                basic_explore.pop(0)
                if(B_indb not in used_indices):
                    in_used = False

            M.update(B_indb)
        t[4] += time.time()-tt

    solutions = np.zeros((len(eff_ind),num_variables))
    x0 = np.zeros(A.shape[1])

    for i,sol in enumerate(solutions):
        sol[eff_ind[i]]=solution_vec[i]

    return eff_ind, solutions