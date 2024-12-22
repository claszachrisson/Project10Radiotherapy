import numpy as np
# import jax
from scipy.optimize import linprog
import time
from collections import deque
from singleSimplex import solveLin
from datetime import datetime

#from tools import LU, tools.ind2bin, tools.bin2ind, tools.Matrices
import tools

results_path = f'results'

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

def find_possible_adjacent(M):

    #Cols with mixed components
    cols = [i for i in range(M.CN_eff.shape[1]) if np.any(M.CN_eff[:, i] > 1e-5) and np.any(M.CN_eff[:, i] < 1e-5)]

    # print(len(cols))
    # print(CN_eff)

    #If no mixed columns no possible eff solution in columns
    if(cols==[]):
        return ()

    #For pivoting
    b_eff = M.Binvb

    #Only consider mixed component columns
    As=np.array(M.BinvN[:,cols])

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
        return ()
    ind = []
    if len(t)>1:
        dominance_matrix = np.all(tC[:, :, None] < tC[:, None, :], axis=0)
        np.fill_diagonal(dominance_matrix, False)
        non_dominated = ~np.any(dominance_matrix, axis=0)

        ind = np.where(non_dominated)[0].tolist()
    else:
        ind=[0]
    
    if ind == []:
        return ()
    
    pivot_ins = index_ins[ind].astype(int)
    pivot_outs = index_outs[ind].astype(int)
    #B = tools.Basis(M.B_indb)
    pivots = []
    for i in range(len(ind)):
        pivots.append( (int(pivot_outs[i]), int(pivot_ins[i])) )
        #pivots.append( (int(pivot_outs[i]), int(M.N_ind[pivot_ins[i]]) ) )
    
    return pivots


def simplex(A,b,C, std_form = True, Initial_basic = None, num_sol = np.inf):
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
    run_datetime = datetime.today().strftime('%d-%m_%H-%M')
    t = [0,0,0,0,0]
    if std_form==False:
        A = np.hstack((A, np.eye(A.shape[0])))
        C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))
    C_row_sum = -C.sum(axis=0)
    depth = -1
    num_variables = len(C[0,:])
    num_basic = len(b) #Number of basic variables is the number of constraints
    num_non_basic = num_variables-num_basic #Number of non-basic is the number of columns minus the basic variables

    #Initial feasible vertex
    if Initial_basic is not None:
        B_ind = Initial_basic
    else:
        B_ind = list(np.arange(num_non_basic, num_variables))
    B_indb = tools.ind2bin(list(B_ind))
    if(B_indb == 0):
        print("Error when converting Initial_basic to binary")
        exit()

    explored_bases = deque(maxlen=1000)
    eff_ind = []
    solution_vec = []
    solutions = []

    def save_solutions(v):
        solution_vec.append(v.Binvb)
        eff_ind.append(v.B_ind.copy())
        sol = np.zeros((1,num_variables))
        sol[v.B_ind]=v.Binvb
        solutions.append(sol)
        #np.savez(f'{results_path}/result_{run_datetime}.npz', 
        #         list_data=eff_ind, 
        #         array_data=solutions)
        if len(eff_ind)==num_sol:
            return -1
        return 1
        
    def step_5(vertex): # Returns (vertex), (eff), (new)
        print("Doing step 5")
        x0 = np.zeros(vertex.AbC.n_vars)
        x0[vertex.B_ind]=vertex.Binvb

        try:
            tlp = time.time()
            result = linprog(C_row_sum, b_ub = -C@x0, A_ub = -C, A_eq = A, b_eq = b, x0=x0, method="revised simplex")
            #result = solveLin(C,A,b,M.B_ind,M.N_ind,x0)
            t[2] += time.time()-tlp
            sol = result.x
        except Exception as e:
            print(f"An error occurred during the linprog optimization: {e}")
            print("Proceeding!")
            return None, False, False

        if np.all(sol == x0):
            return vertex, True, False
        else:
            tmp_B_ind = list(np.where(np.abs(sol)>1e-6)[0])
            tmp_B_indb = tools.ind2bin(tmp_B_ind)
            if (len(tmp_B_ind)==num_basic) and not tmp_B_indb in explored_bases:
                v = tools.Vertex()
                v.new(tmp_B_indb,vertex.AbC)
                return v, True, True
        return None, False, False
    
    def _simplex(vertex, depth): # Main recursion function
        depth+=1
        t[4] = time.time() - tstart
        print(f"Recursion depth {depth}, with solution len {len(eff_ind)}")
        print(f"Times spent: {[round(tt,3) for tt in t]}")
        explored_bases.append(vertex.B_indb)
        tt = time.time()
        eff = check_efficient(vertex)
        t[0] += time.time()-tt
        if not eff:
            tt = time.time()
            vertex, eff, new = step_5(vertex)
            t[1] += time.time() - tt
            if new:
                explored_bases.append(vertex.B_indb)
        if eff:
            print("Efficient solution found.")
            #first_sol = False
            if save_solutions(vertex) == -1:
                return -1
            pivots = find_possible_adjacent(vertex)
            for p in pivots:
                B_indb = (vertex.B_indb & ~(1 << vertex.B_ind[p[0]])) | (1 << vertex.N_ind[p[1]])
                if B_indb not in explored_bases:
                    adjacent = tools.Vertex()
                    adjacent.pivot(vertex, p, B_indb)
                    #print(f"The gods decided {adjacent.B_ind} was not in explored_bases")
                    #s = _simplex(adjacent)
                    if _simplex(adjacent,depth) == -1:
                        return -1
            print("pivots returned")
            return 0
    
    print("STARTING MOLP SIMPLEX")
    print(f"# Vars: {num_variables}; # Basic: {num_basic}")
    #Create the matrices for the initial solution
    AbC = tools.MOLPP(A,b,C,num_variables,num_basic)
    v = tools.Vertex()
    v.new(B_indb, AbC)

    ideal_sol = np.all(v.CN_eff<=0)
    if(ideal_sol==True):
        return True
    
    # Begin recursion
    tstart = time.time()
    s = _simplex(v,depth)
    print("MOLP SIMPLEX STOPPED")
    if num_sol < np.inf and s != -1:
        print("Something went wrong and _simplex() stopped early.")
        return eff_ind, solutions
    
    print(f"MOLP SIMPLEX STOPPED at {num_sol}")
    return eff_ind, solutions