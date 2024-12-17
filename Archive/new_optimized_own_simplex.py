import numpy as np
# import jax
from scipy.linalg import LinAlgWarning
from scipy.optimize import linprog
from LUsolve import LUsolve
from singleSimplex import solve
import warnings
import random


def check_efficient(B_inv,CN,CB,N,first_sol,C,b, basic_ind):
    
    #Reduced cost vector
    CN_eff = CN-CB@B_inv.solve(N)

    #Check if any row is positive non-zero
    pos_nonzero = np.any(np.logical_or(CN_eff < 0, np.all(CN_eff == 0, axis=0)), axis=0) #Check if any column is all negative or if all zero (check correctness)

    #If positive non-zero the solution is not efficient
    if(np.any(pos_nonzero==False)):
        print("pos_nonzero")
        return False

    #If the first solution check if there is an ideal solution
    if first_sol:
        ideal_sol = np.all(CN_eff<=0)
        if(ideal_sol==True):
            return True
    
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

def find_all_eff_sol(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies,b):
    #
    As = B_inv.solve(N)

    #Reduced cost vector
    CN_eff = CN-CB@As

    #Cols with mixed components
    cols = [i for i in range(CN_eff.shape[1]) if np.any(CN_eff[:, i] > 0) and np.any(CN_eff[:, i] < 0)]

    #
    if(cols==[]):
        print("No mixed component columns")
        return [], []


    # print("hej",cols)

    
    b_eff = B_inv.solve(b)

    As=np.array(As[:,cols])

    

    t = np.full(len(As[0,:]),np.inf)
    index_ins, index_outs = np.full(len(As[0,:]),np.inf), np.full(len(As[0,:]),np.inf)
    for s in range(len(cols)):
        valid = As[:, s] > 0
        values = np.where(valid, b_eff / As[:, s], np.inf)
        min_index = np.argmin(values)
        t[s] = values[min_index]
        if t[s] < np.inf:
            index_outs[s] = min_index
            index_ins[s] = cols[s]
    
    # print(len(index_outs),len(cols))
    # ind_to_remove=[]
    # # print(cols)
    # for i in range(len(index_outs)):

    #     pivot_in = int(index_ins[i])
    #     pivot_out = int(index_outs[i])

    #     temp_basic_ind = basic_ind.copy()
    #     temp_basic_ind[pivot_out] = non_basic_ind[pivot_in]
    #     # print(temp_basic_ind)

    #     # temp_basic_ind=np.array([3,4,5])
    #     # print(used_indicies)

    #     if any(np.array_equal(temp_basic_ind, used) for used in used_indicies):
    #         ind_to_remove.append(i)
    #         # cols.pop(i)
    #         # print(cols)
    #     # print(ind_to_remove)
    # for i in reversed(ind_to_remove):
    #     cols.pop(i)  # Pop the element at index i
    # print(cols)
    tC = np.zeros((len(CN_eff[:,0]),len(cols)))
    for i in range(len(cols)):
        tC[:,i]=t[i]*CN_eff[:,cols[i]]

    # np.all(tC[:,i])
    # print(np.array(tC))
    # print(cols)
    if cols==[]:
        print("Should never be printed?")
        return [], []
    ind = []
    print(cols)
    if len(cols)>1:
        dominance_matrix = np.all(tC[:, :, None] <= tC[:, None, :], axis=0)
        np.fill_diagonal(dominance_matrix, False)
        non_dominated = ~np.any(dominance_matrix, axis=0)

        # Indices of non-dominated columns
        ind = np.where(non_dominated)[0].tolist()
    else:
        ind=[0]

    if ind == []:
        print("No ind tsCs>trCr")
        return [], []
    

    
    # print(index_ins[ind])
    # print("HEJ")
    # print(ind) 
    # print(tC[:,1])
    non_basic_ind_list = list(np.zeros((len(ind),(len(non_basic_ind))),dtype=int))
    basic_ind_list = list(np.zeros((len(ind),(len(basic_ind))),dtype=int))

    print(ind,index_ins)
    print(index_outs)
    print("Basic ind", basic_ind)

    for i,index in enumerate(ind):
        pivot_in = int(index_ins[index])
        pivot_out = int(index_outs[index])
        # print(pivot_in,pivot_out)

        

        tmp_non_basic = non_basic_ind.copy()
        tmp_basic = basic_ind.copy()

        tmp = tmp_non_basic[pivot_in]

        tmp_non_basic[pivot_in] = basic_ind[pivot_out]
        tmp_basic[pivot_out] = tmp
        # print(index_ins, index_outs, t)
        tmp = non_basic_ind[pivot_in]
        non_basic_ind_list[i] = tmp_non_basic

        basic_ind_list[i] = sorted(tmp_basic)
        # print("hej",tmp_basic)
        print(basic_ind_list[i])

    # print(non_basic_ind_list)
    print("Should be sorted",basic_ind_list)
    # used_indicies.append(basic_ind.copy())
    # print(np.max(CN_eff[0,cols]))
    # print(CN_eff)


    
    # for i in cols:

    # print(basic_ind,non_basic_ind)




    # print(cols)

    return basic_ind_list,non_basic_ind_list

def find_first_eff_sol(non_basic_ind, basic_ind, B_inv, N, used_indicies,b):
    # index=9999 #remove
    As = np.copy(N)
    # print(B_inv)
    # print(N)x
    As = B_inv.solve(N)
    b_eff = B_inv.solve(b)
    # print(b_eff)


    positive_in_columns = np.any(As > 0, axis=0)
    # print(positive_in_columns)

    As = As[:, positive_in_columns]


    # print(As)

    # As[2,0]=-1
    # for i in range(len(non_basic_ind)):
    #     neg_col = np.any(As[:, i] < 0) #Check if column is negative (Not correct)
    #     if(neg_col):
    #         pivot_ind = non_basic_ind[i]
    #         #Here add how to switch
    #         break
    # H

    # print(As)
    # if(neg_col==False):
    min_value = np.inf
    index_in, index_out = np.inf, np.inf
    # print([used for used in used_indicies])
    # print(any(np.array_equal([3,4,2], used) for used in used_indicies))
    for i in range(len(b_eff)):
        for s in range(len(As[0,:])):
            temp_basic_ind = basic_ind.copy()
            temp_non_basic_ind = non_basic_ind.copy()
            temp_basic_ind[i] = non_basic_ind[s]
            temp_basic_ind.sort()
            temp_non_basic_ind[s] = basic_ind[i]
            # print(temp_basic_ind)
            if temp_basic_ind in used_indicies:
                # print(temp_basic_ind, used_indicies)
                continue
            if As[i,s]>0:
                value=b_eff[i]/As[i,s]
            else:
                value = np.inf
            # print(value)
            
            if(value<min_value):
                # print(As[i,s])
                # print(value)
                min_value = value
                index_out = i
                index_in = s


    # print("Basic and non basic")
    if index_in==np.inf or index_out==np.inf:
        index_in = random.randint(0,len(As[0,:])-1)
        index_out = random.randint(0,len(b_eff)-1)
        used_indicies.append(basic_ind.copy())
        # raise ValueError(basic_ind,non_basic_ind,used_indicies) 
        # return [], []

       
    # tmp = non_basic_ind[s]
    # non_basic_ind[s] = basic_ind[i]
    # basic_ind[i] = tmp

    tmp = non_basic_ind[index_in]
    non_basic_ind[index_in] = basic_ind[index_out]
    basic_ind[index_out] = tmp

    basic_ind.sort()
    used_indicies.append(basic_ind.copy())
    # print(used_indicies)
    # print(min_value,(index_in,index_out))
    # print(basic_ind,non_basic_ind)

    # print(basic_ind)
    

    # print(As)
    # print(neg_col)
    # indices = feasible.nonzero()
    # print(indices)
    # if(A)
    # bi=B[:,basic_ind]
    
    # ts=np.min()
    # print(As)
    return basic_ind,non_basic_ind

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
    # print(basic_ind)

    #Create the matrices for the initial solution
    B = A[:,basic_ind]
    N = A[:,non_basic_ind]
    CN = C[:,non_basic_ind]
    CB = C[:,basic_ind]
    B_inv = LUsolve(B)


    first_sol = True
    eff = check_efficient(B_inv,CN,CB,N,first_sol,C,b, basic_ind) #Check efficiency of the initial solution

    #Loop to find initial efficient solution

    while not eff:

        basic_ind,non_basic_ind=find_first_eff_sol(non_basic_ind, basic_ind, B_inv, N, used_indicies,b) #Pivot to try to find first solution
        
        #Update matrices according to the pivot
        basic_ind.sort()
        B = A[:,basic_ind]
        N = A[:,non_basic_ind]
        CN = C[:,non_basic_ind]
        CB = C[:,basic_ind]
        print("U")
        print(basic_ind)
        # print(B)
        # B_inv=np.linalg.inv(B) #Can use try for the case that the matrix is singular
        try:
            # B_inv = np.linalg.pinv(B)
            B_inv = LUsolve(B)
            # print(B)
            # print(B_inv)
        except LinAlgWarning:
            print("SINGULAR MATRIX!")
            basic_ind = used_indicies[-2]
            print(basic_ind)
            non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]
            print(non_basic_ind)
            B = A[:,basic_ind]
            N = A[:,non_basic_ind]
            CN = C[:,non_basic_ind]
            CB = C[:,basic_ind]
            continue
        eff = check_efficient(B_inv,CN,CB,N,first_sol,C,b,basic_ind) #Check efficiency of the solution after the pivot
        if eff:
            # print("Basic ind",basic_ind)
            # print("Non basic ind",non_basic_ind)
            # print(CN-CB@B_inv@N)
            # print(CN)
            # print(CB)
            # print(B_inv)
            # print("N",N)
            # print("B", B)
            print("check efficient",basic_ind)
            solution_vec.append(B_inv.solve(b))        
            eff_ind.append(basic_ind.copy())
        else:
            C_row_sum = -C.sum(axis=0)
            x0 = np.zeros(A.shape[1])
            try:
                x0[basic_ind]=B_inv.solve(b)
            except LinAlgWarning:
                continue
            print(x0)

            result = sorted(solve(C,A,b,B,basic_ind,non_basic_ind,x0))
            check_arr = np.arange(num_basic+num_non_basic,num_basic+num_non_basic+C.shape[0])
            if result == [] or not np.array_equal(check_arr,result[:-C.shape[0]]):
                print("IN THE WEIRD CHECK", result)
                continue
            print("LINPROG", result)
            result = result[:-C.shape[0]]
            # print()
            solution_vec.append(B_inv.solve(b))
            eff_ind.append(basic_ind.copy())
            used_indicies.append(basic_ind.copy())

            # if(len(tmp_basic_ind)>len(basic_ind)):
            #     print("AAAAAAA",tmp_basic_ind)
            print("FOUND SOL", result)
            eff = True
            basic_ind=sorted(result)
            non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]


            B = A[:,basic_ind]
            N = A[:,non_basic_ind]
            CN = C[:,non_basic_ind]
            CB = C[:,basic_ind]
            B_inv=LUsolve(B)

                # solution_vec.append(B_inv@b)
                # eff_ind.append(basic_ind.copy())
                
            # print("HEJ2")
            # print(basic_ind)
            # print(CN-CB@B_inv@N)
            # print(B_inv@b)
    # print(eff_ind)
    # print(solution_vec)
    # Cx = CB@B_inv@b
    # basic_ind,non_basic_ind=find_all_eff_sol(non_basic_ind, basic_ind, B_inv, N, used_indicies)
    # eff = check_efficient(B_inv,CN,CB,b,N)
    print(eff_ind)
    first_sol = False
    sols=True
    basic_explore = []
    non_basic_explore = []
    while sols:

        basic_ind_list,non_basic_ind_list=find_all_eff_sol(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies,b)
        print(basic_explore)
        print("HEJ",used_indicies)
        if np.shape(basic_ind_list)!=(0,):
            for basic, non_basic in zip(basic_ind_list, non_basic_ind_list):
                if any(np.array_equal(basic, used) for used in used_indicies):
                    continue
                else:
                    used_indicies.append(basic.copy())
                    basic_explore.append(basic.copy())
                    non_basic_explore.append(non_basic.copy())
        if len(basic_explore)==0:
            sols = False
        
        # print(basic_explore)
        # print(non_basic_explore)
        # print(used_indicies)
        else:
            print("This index will be checked", basic_explore[0])
            basic_ind = basic_explore[0]
            non_basic_ind = non_basic_explore[0]
            basic_explore.pop(0)
            non_basic_explore.pop(0)
            
            # print(basic_ind,non_basic_ind)


            B = A[:,basic_ind]
            N = A[:,non_basic_ind]
            CN = C[:,non_basic_ind]
            CB = C[:,basic_ind]
            B_inv=LUsolve(B) #Can use try for the case that the matrix is singular
            # print(basic_ind)
        eff = check_efficient(B_inv,CN,CB,N,first_sol,C,b,basic_ind)
        if eff:
            # print(eff_ind)
            # print("Basic ind",basic_ind)
            # print("Non basic ind",non_basic_ind)
            # print(CN-CB@B_inv@N)
            # print(CN)
            # print(CB)
            # print(B_inv)
            # print("N",N)
            # print("B", B)
            if not any(np.array_equal(basic_ind, explore) for explore in eff_ind):
                solution_vec.append(B_inv.solve(b))
                eff_ind.append(basic_ind.copy())
        else:
            C_row_sum = -C.sum(axis=0)
            x0 = np.zeros(A.shape[1])

            x0[basic_ind]=B_inv.solve(b)

            result = sorted(solve(C,A,b,B,basic_ind,non_basic_ind,x0))
            check_arr = np.arange(num_basic+num_non_basic,num_basic+num_non_basic+C.shape[0])
            print("check array",check_arr)
            
            if result == [] or not np.array_equal(check_arr,result[:-C.shape[0]]):
                continue
            print("LINPROG", result)
            result = result[:-C.shape[0]]
            print("LINPROG", result)
            if np.all(result == basic_ind):
                print("FOUND SOL", basic_ind)
                if not any(np.array_equal(basic_ind, explore) for explore in eff_ind):
                    solution_vec.append(B_inv.solve(b))
                    eff_ind.append(basic_ind.copy())

            else:

                if any(np.array_equal(result, explore) for explore in basic_explore):
                    solution_vec.append(B_inv.solve(b))
                    eff_ind.append(result.copy())
                    basic_explore.remove(result.copy())
                    basic_ind = sorted(result.copy())
                    non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]
                    B = A[:,basic_ind]
                    N = A[:,non_basic_ind]
                    CN = C[:,non_basic_ind]
                    CB = C[:,basic_ind]
                    B_inv=LUsolve(B)
                else:

                        
            # if(len(tmp_basic_ind)<len(basic_ind)):
            #     tmp_basic_ind.append()

                    if any(np.array_equal(result, used) for used in used_indicies):
                        
                        # if any(np.array_equal(tmp_basic_ind, used) for used in used_indicies):
                        #     continue
                        # else:
                        
                        basic_ind = sorted(result)
                        non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]

                        #Add new non-basic list
                        #Also check if the basic list is already explored

                        B = A[:,basic_ind]
                        N = A[:,non_basic_ind]
                        CN = C[:,non_basic_ind]
                        CB = C[:,basic_ind]
                        print(basic_ind)
                        print(b)
                        # B_inv = np.linalg.pinv(B)
                        B_inv = LUsolve(B)
                        # print(B)
                        # print(B_inv)
                        if not any(np.array_equal(basic_ind, explore) for explore in eff_ind):
                            solution_vec.append(B_inv.solve(b))
                            eff_ind.append(basic_ind.copy())
                            used_indicies.append(basic_ind.copy())

                # solution_vec.append(B_inv@b)
                # eff_ind.append(basic_ind.copy())
            
            # print("HEJ3")
            # print(basic_ind)
            # print(CN-CB@B_inv@N)
            # print(CN)
    # print(used_indicies)
    # print("HEJ")
    # print(eff_ind)

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

