import numpy as np
# import jax
from scipy.optimize import linprog
from LUsolve import LUsolve


def check_efficient(B_inv,CN,CB,N):
    
    #Reduced cost vector
    CN_eff = CN-CB@B_inv.solve(N)

    #Check if any row is positive non-zero
    pos_nonzero = np.any(np.logical_or(CN_eff < 0, np.all(CN_eff == 0, axis=0)), axis=0) #Check if any column is all negative or if all zero (check correctness)

    #If positive non-zero the solution is not efficient
    if(np.any(pos_nonzero==False)):
        print("pos_nonzero")
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

def find_possible_eff_sols(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies,b):

    As = B_inv.solve(N)

    #Reduced cost vector
    CN_eff = CN-CB@As

    #Cols with mixed components
    cols = [i for i in range(CN_eff.shape[1]) if np.any(CN_eff[:, i] > 0) and np.any(CN_eff[:, i] < 0)]

    #
    if(cols==[]):
        print("No mixed component columns")
        return []



    #For pivoting
    b_eff = B_inv.solve(b)

    #Only consider mixed component columns
    As=np.array(As[:,cols])

    

    t = np.full(len(As[0,:]),np.inf)
    index_ins, index_outs = np.full(len(As[0,:]),np.inf), np.full(len(As[0,:]),np.inf)
    for s in range(len(cols)):
        valid = As[:, s] > 0
        values = np.where(valid, b_eff / As[:, s], np.inf)
        min_index = np.argmin(values)
        t[s] = values[min_index]
        if t[s] < np.inf:
            index_outs[s] = int(min_index)
            index_ins[s] = int(cols[s])
    
    tC = np.zeros((len(CN_eff[:,0]),len(cols)))
    for i in range(len(cols)):
        tC[:,i]=t[i]*CN_eff[:,cols[i]]

    if cols==[]:
        print("Should never be printed?")
        return []
    ind = []
    if len(cols)>1:
        dominance_matrix = np.all(tC[:, :, None] <= tC[:, None, :], axis=0)
        np.fill_diagonal(dominance_matrix, False)
        non_dominated = ~np.any(dominance_matrix, axis=0)

        ind = np.where(non_dominated)[0].tolist()
    else:
        ind=[0]

    if ind == []:
        print("No ind tsCs>trCr")
        return []
    

    tmp_basic_ind_list = np.tile(basic_ind, (len(ind), 1))


    rows = np.arange(len(ind))
    pivot_ins = index_ins[ind].astype(int)
    pivot_outs = index_outs[ind].astype(int)


    tmp_non_basic_ind = np.array(non_basic_ind.copy())

    tmp_basic_ind_list[rows, pivot_outs] = tmp_non_basic_ind[pivot_ins]

    tmp_basic_ind_list = np.sort(tmp_basic_ind_list, axis=1)
    
    used_indicies_tmp = np.array(used_indicies.copy())

    matches = np.any(np.all(tmp_basic_ind_list[:, None, :] == used_indicies_tmp[None, :, :], axis=2),axis=1)
    basic_ind_list = tmp_basic_ind_list[~matches].tolist()

    return basic_ind_list


def simplex(A,b,C, num_sol = 100):
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

    num_basic = len(b) #Number of basic variables is the number of constraints
    num_non_basic = len(C[0,:])-num_basic #Number of non-basic is the number of columns minus the basic variables
    
    #Initial feasible solution
    basic_ind = list(np.arange(num_non_basic, num_basic + num_non_basic))
    non_basic_ind = list(np.arange(0,num_non_basic))

    #Create a vector saving for saving the results
    used_indicies = [basic_ind.copy()]
    eff_ind = []
    solution_vec = []

    #Create the matrices for the initial solution
    B = A[:,basic_ind]
    N = A[:,non_basic_ind]
    CN = C[:,non_basic_ind]
    CB = C[:,basic_ind]
    B_inv = LUsolve(B)

    #Loop to find initial efficient solution

    first_sol = False
    sols=True
    basic_explore = []
    CN_eff = CN-CB@B_inv.solve(N)

    ideal_sol = np.all(CN_eff<=0)
    if(ideal_sol==True):
        return True
    
    while sols:
        eff = check_efficient(B_inv,CN,CB,N)
        if eff:

            solution_vec.append(B_inv.solve(b))
            eff_ind.append(basic_ind.copy())
            used_indicies.append(basic_ind.copy())

        else:
            C_row_sum = -C.sum(axis=0)
            x0 = np.zeros(A.shape[1])

            x0[basic_ind]=B_inv.solve(b)

            result = linprog(C_row_sum, b_ub = -C@x0, A_ub = -C, A_eq = A, b_eq = b, x0=x0, method="revised simplex")
            sol = result.x
            tmp_basic_ind = np.where(np.abs(sol)>1e-6)
            tmp_basic_ind = list(tmp_basic_ind[0])
            if len(tmp_basic_ind)==len(basic_ind) and not any(np.array_equal(tmp_basic_ind, used) for used in used_indicies):

                basic_ind = sorted(tmp_basic_ind)
                non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]

                

                #Add new non-basic list
                B = A[:,basic_ind]
                N = A[:,non_basic_ind]
                CN = C[:,non_basic_ind]
                CB = C[:,basic_ind]
                B_inv = LUsolve(B)

                solution_vec.append(B_inv.solve(b))
                eff_ind.append(basic_ind.copy())
                used_indicies.append(basic_ind.copy())



        basic_ind_list = find_possible_eff_sols(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies,b)

        for basic in basic_ind_list:
            used_indicies.append(basic.copy())
            basic_explore.append(basic.copy())
        if len(basic_explore)==0:
            sols = False
        
        else:
            print(len(basic_explore))
            print(basic_explore)
            print("This index will be checked", basic_explore[0])
            basic_ind = basic_explore[0]
            non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]

            basic_explore.pop(0)            


            B = A[:,basic_ind]
            N = A[:,non_basic_ind]
            CN = C[:,non_basic_ind]
            CB = C[:,basic_ind]
            B_inv=LUsolve(B)


    solutions = np.zeros((len(eff_ind),num_non_basic+num_basic))
    x0 = np.zeros(A.shape[1])

    for i,sol in enumerate(solutions):
        sol[eff_ind[i]]=solution_vec[i]


    # print(solutions)
    print(eff_ind)
    print(solutions)
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

A = np.array([[1,2,1,1,2,1,2],[-2,-1,0,1,2,0,1],[0,1,2,-1,1,-2,-1]])
A = np.hstack((A, np.eye(A.shape[0])))
# print(A)
# print(A)
b = np.array([16,16,16])
C = np.array([[1,2,-1,3,2,0,1],[0,1,1,2,3,1,0],[1,0,1,-1,0,-1,-1]])
C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))

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

simplex(A,b,C)
