import numpy as np
import scipy.sparse as sci_sp
# import jax
from scipy.optimize import linprog

from tools import LU
from in_reference import is_row_in_reference
from singleSimplex import solveLin


def check_efficient(B_inv,CN,CB,N):
    
    #Reduced cost vector
    CN_eff = CN-CB@B_inv.solve(N)

    #Check if any row is positive non-zero
    pos_nonzero = np.any(np.logical_or(CN_eff < 0, np.all(CN_eff == 0, axis=0)), axis=0) #Check if any column is all negative or if all zero (check correctness)

    #If positive non-zero the solution is not efficient
    if(np.any(pos_nonzero==False)):
        return False
    
    #If any row is negative the solution is efficient
    any_row_negative = np.all(CN_eff < 0, axis=1)
    if(np.any(any_row_negative==True)):
        return True
    
    #Calculate the sum of the rows (sum the columns)
    row_sums = CN_eff.sum(axis=0)

    #If all are 0 or negative (some small tolerance)
    if(np.all(row_sums<=1e-6)):
        return True
    
    return False

def find_possible_eff_sols(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies, basic_explore, b, eff_ind):

    As = B_inv.solve(N)

    #Reduced cost vector
    CN_eff = CN-CB@As

    #Cols with mixed components
    cols = [i for i in range(CN_eff.shape[1]) if np.any(CN_eff[:, i] > 1e-6) and np.any(CN_eff[:, i] < 1e-6)]

    # print(len(cols))
    # print(CN_eff)

    #If no mixed columns no possible eff solution in columns
    if(cols==[]):
        return []

    

    #For pivoting
    b_eff = B_inv.solve(b)

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
    tC = t * CN_eff[:, cols]
    # print(tC)
    if cols.size==0:
        return []
    ind = []
    if len(t)>1:
        dominance_matrix = np.all(tC[:, :, None] < tC[:, None, :], axis=0)
        np.fill_diagonal(dominance_matrix, False)
        non_dominated = ~np.any(dominance_matrix, axis=0)

        ind = np.where(non_dominated)[0].tolist()
        # print(ind)
    else:
        ind=[0]
    
    if ind == []:
        return []
    

    tmp_basic_ind_list = np.tile(basic_ind, (len(ind), 1))

    # Use the indices from index_ins and index_outs
    rows = np.arange(len(ind))
    pivot_ins = index_ins[ind].astype(int)
    pivot_outs = index_outs[ind].astype(int)
    # print(pivot_ins)
    # print(basic_ind[pivot_outs])
    # print(pivot_ins)

    # Create a temporary copy of non_basic_ind
    tmp_non_basic_ind = np.array(non_basic_ind.copy())
    
    # print(tmp_basic_ind_list[pivot_ins])
    # Perform the swap
    tmp_basic_ind_list[rows, pivot_outs] = tmp_non_basic_ind[pivot_ins]


    # Sort the rows of tmp_basic_ind_list
    tmp_basic_ind_list = np.sort(tmp_basic_ind_list, axis=1)

    # Assuming used_indicies is a 2D array where each row is a set of used indices
    used_indicies_tmp = np.array(used_indicies.copy())
    basic_explore_tmp = np.array(basic_explore.copy())

    if used_indicies_tmp.size > 0:
        matches_in_used = is_row_in_reference(tmp_basic_ind_list, used_indicies_tmp)
    else:
        matches_in_used = np.zeros(tmp_basic_ind_list.shape[0], dtype=bool)
    if len(basic_explore) > 0:
        matches_in_explore = is_row_in_reference(tmp_basic_ind_list, basic_explore_tmp)
    else:
        matches_in_explore = np.zeros(tmp_basic_ind_list.shape[0], dtype=bool)
    #TRY TO OPTIMIZE THIS
    # 
    #     # Search for each row in tmp_basic_ind_list_sorted in the sorted used_indicies_tmp
        
    #     matches_in_used = np.array([np.any(np.all(row == used_indicies_tmp, axis=1)) for row in tmp_basic_ind_list])
    # else:
    #     matches_in_used = np.zeros(tmp_basic_ind_list.shape[0], dtype=bool)

    # if len(basic_explore) > 0:
    #     # Search for each row in tmp_basic_ind_list_sorted in the sorted basic_explore
    #     matches_in_explore = np.array([np.any(np.all(row == basic_explore, axis=1)) for row in tmp_basic_ind_list])
    # else:
    #     matches_in_explore = np.zeros(tmp_basic_ind_list.shape[0], dtype=bool)
    # Combine the matches: True if the row is in either used_indicies_tmp or basic_explore
    matches_in_both = matches_in_used | matches_in_explore

    # matches_in_both = matches_in_explore

    # Remove the rows that match in both matrices
    basic_ind_list = tmp_basic_ind_list[~matches_in_both].tolist()

    # Remove matching rows from tmp_basic_ind_list
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
    

    num_basic = len(b) #Number of basic variables is the number of constraints
    num_non_basic = len(C[0,:])-num_basic #Number of non-basic is the number of columns minus the basic variables

    #Initial feasible solution
    if Initial_basic is not None:
        basic_ind = Initial_basic
    else:
        basic_ind = list(np.arange(num_non_basic, num_basic + num_non_basic))

    non_basic_ind = list(np.arange(0,num_non_basic))

    #Create a vector saving for saving the results
    used_indicies = []
    eff_ind = []
    solution_vec = []

    #Create the matrices for the initial solution
    B = A[:,basic_ind]
    N = A[:,non_basic_ind]
    CN = C[:,non_basic_ind]
    CB = C[:,basic_ind]
    B_inv = LU(B)

    #Loop to find initial efficient solution

    first_sol = False
    sols=True
    basic_explore = []
    CN_eff = CN-CB@B_inv.solve(N)

    ideal_sol = np.all(CN_eff<=0)
    if(ideal_sol==True):
        return True
    
    while sols:
        iters+=1
        print(f"Iteration {iters}, with solution len {len(eff_ind)}, basic explore: {len(basic_explore)}")
        eff = check_efficient(B_inv,CN,CB,N)
        if eff:
            print("CHECK EFF")
            solution_vec.append(B_inv.solve(b))
            eff_ind.append(basic_ind.copy())
            # used_indicies.append(basic_ind.copy())

        else:
            C_row_sum = -C.sum(axis=0)
            x0 = np.zeros(A.shape[1])

            # print(B_inv.solve(b).shape)

            x0[basic_ind]=B_inv.solve(b)

            try:
                sol = solveLin(C,A,b,basic_ind,non_basic_ind,x0)
            except Exception as e:
                print(f"An error occurred during the linprog optimization: {e}")
                print("Proceeding!")
                sol=np.zeros(x0.shape)

            if np.all(sol == x0):
                tmp_basic_ind = basic_ind.copy()
            else:
                tmp_basic_ind = np.where(np.abs(sol)>1e-6)
                tmp_basic_ind = list(tmp_basic_ind[0])
                
            # print("LINPROG",tmp_basic_ind)

            if len(tmp_basic_ind)==len(basic_ind) and not any(np.array_equal(tmp_basic_ind, used) for used in used_indicies) and not np.any(sol<-1e-6):
                print("LINPROG")
                basic_ind = sorted(tmp_basic_ind)
                non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]

                

                #Add new non-basic list
                B = A[:,basic_ind]
                N = A[:,non_basic_ind]
                CN = C[:,non_basic_ind]
                CB = C[:,basic_ind]
                B_inv = LU(B)

                solution_vec.append(B_inv.solve(b))
                eff_ind.append(basic_ind.copy())
                # used_indicies.append(basic_ind.copy())



        
        basic_ind_list = find_possible_eff_sols(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies,basic_explore,b, eff_ind)
        print(len(basic_ind_list))
        for basic in basic_ind_list:
        
            basic_explore.append(basic.copy())
        if len(basic_explore)==0:
            sols = False
        
        else:

            used_indicies.append(basic_ind.copy())
            basic_ind = basic_explore[0]
            
            non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]

            basic_explore.pop(0)            


            B = A[:,basic_ind]
            N = A[:,non_basic_ind]
            CN = C[:,non_basic_ind]
            CB = C[:,basic_ind]
            B_inv=LU(B)


    solutions = np.zeros((len(eff_ind),num_non_basic+num_basic))
    x0 = np.zeros(A.shape[1])

    for i,sol in enumerate(solutions):
        sol[eff_ind[i]]=solution_vec[i]

    print(used_indicies)
    np.savez('result.npz', list_data=eff_ind, array_data=solutions)
    return eff_ind, solutions

    # print(ind)



# A = np.array([[-1,-2,0],[-1,0,2],[1,0,-1]])
# A = np.hstack((A, np.eye(A.shape[0])))
# b = np.array([1,2,4])
# C = np.array([[1,1,0],[0,1,0],[1,-1,1]])
# C = np.hstack((C, np.eye(C.shape[0])))

# A = np.array([[0,1,4],[1,1,0]])
# A = np.hstack((A, np.eye(A.shape[0])))
# # print(A)
# b = np.array([8,8])
# C = np.array([[0,-2,1],[-1,2,1]])
# C = np.hstack((C, np.zeros((A.shape[0], A.shape[0]))))
# print(C)

# A = np.array([[2,3,1],[4,1,2],[3,4,2]])
# A = np.hstack((A, np.eye(A.shape[0])))
# # print(A)
# b = np.array([5,11,8])
# C = np.array([[5,4,3],[3,1,-1]])
# C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))

# A = np.array([[1,1,2],[1,2,1],[2,1,1]])
# A = np.hstack((A, np.eye(A.shape[0])))
# # print(A)
# b = np.array([12,12,12])
# C = np.array([[6,4,5],[0,0,1]])
# C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))

# A = np.array([[1,2,1,1,2,1,2],[-2,-1,0,1,2,0,1],[0,1,2,-1,1,-2,-1]])
# A = np.hstack((A, np.eye(A.shape[0])))
# # print(A)
# # print(A)
# b = np.array([16,16,16])
# C = np.array([[1,2,-1,3,2,0,1],[0,1,1,2,3,1,0],[1,0,1,-1,0,-1,-1]])
# C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))

# A = np.array([[1,1,0],[0,1,0],[1,-1,1]])
# A = np.hstack((A, np.eye(A.shape[0])))
# # print(A)
# b = np.array([1,2,4])
# C = np.array([[-1,-2,0],[-1,0,2],[1,0,-1]])
# C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))

# A = np.array([[-6,3],[3,2],[4,-2],[0,1]])
# A = np.hstack((A, np.eye(A.shape[0])))
# print(A)
# # print(A)
# b = np.array([12,30,24,6])
# C = np.array([[9,6],[-3,2]])
# C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))

# indices, result = simplex(A,b,C)
# print(indices)


