import numpy as np
#import jax
from scipy.optimize import linprog
from LUsolve import *
# pivot_column = A[:, s]

# # Calculate the ratios b_i / a_i^s where a_i^s > 0
# ratios = b / pivot_column

# # Only consider rows where pivot_column > 0 (i.e., a_i^s > 0)
# positive_ratios = ratios[pivot_column > 0]

# # Calculate the minimum ratio
# t_s = np.min(positive_ratios)

def check_efficient(B_inv,CN,CB,N,first_sol,C,b, basic_ind):
    
    #CN_eff = CN-CB@B_inv@N
    CN_eff = CN-CB@B_inv.solve(N)
    # print(np.shape(CN_eff))
    # N_eff = B_inv@N
    # Cx = CB@B_inv@b

    pos_nonzero = np.any(np.logical_or(CN_eff < 0, np.all(CN_eff == 0, axis=0)), axis=0) #Check if any column is all negative or if all zero (check correctness)
    # print(pos_nonzero)
    # print(CN_eff)
    if(np.any(pos_nonzero==False)):
        print("pos_nonzero")
        return False

    if first_sol:
        ideal_sol = np.all(CN_eff<=0)
        if(ideal_sol==True):
            print("Ideal solution")
            return True
    
    any_row_negative = np.all(CN_eff < 0, axis=1)
    if(np.any(any_row_negative==True)):
        print("Any row negative")
        # print(CN_eff)
        return True
    
    row_sums = CN_eff.sum(axis=0)

    print("Row sum", row_sums)
    if np.all(basic_ind==[2,4,8]):
        print("[2,4,8]!!!!!",CN_eff)
    if(np.all(row_sums<=1e-6)):
        print("Row sums")
        print(row_sums)
        print(CN_eff)
        return True
    

    return False

def find_all_eff_sol(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies):
    CN_eff = CN-CB@B_inv.solve(N)
    # print(CN_eff)
    print(CN_eff.shape[1])
    cols = [i for i in range(CN_eff.shape[1]) if np.any(CN_eff[:, i] > 0) and np.any(CN_eff[:, i] < 0)]

    if(cols==[]):
        print("No mixed component columns")
        return [], []


    # print("hej",cols)

    As = B_inv.solve(N)
    b_eff = B_inv.solve(b)

    As=np.array(As[:,cols])

    

    t = np.full(len(As[0,:]),np.inf)
    index_ins, index_outs = np.full(len(As[0,:]),np.inf), np.full(len(As[0,:]),np.inf)
    for i in range(len(b_eff)):
        for s in range(len(cols)):
            # print(temp_basic_ind)
                # print(temp_basic_ind, used_indicies)
            if As[i,s]>0:
                value=b_eff[i]/As[i,s]
            else:
                value = np.inf
            # print(value)
            
            if(value<t[s]):
                # print(As[i,s])
                # print(value)
                t[s] = value
                index_outs[s] = i
                index_ins[s] = cols[s]
    
    # print(len(index_outs),len(cols))
    ind_to_remove=[]
    # print(cols)
    for i in range(len(index_outs)):

        pivot_in = int(index_ins[i])
        pivot_out = int(index_outs[i])

        temp_basic_ind = basic_ind.copy()
        temp_basic_ind[pivot_out] = non_basic_ind[pivot_in]
        # print(temp_basic_ind)

        # temp_basic_ind=np.array([3,4,5])
        # print(used_indicies)

        if any(np.array_equal(temp_basic_ind, used) for used in used_indicies):
            ind_to_remove.append(i)
            # cols.pop(i)
            # print(cols)
        # print(ind_to_remove)
    for i in reversed(ind_to_remove):
        cols.pop(i)  # Pop the element at index i
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
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    # print(i,j)
                    if np.all(tC[:,j]>=tC[:,i]):
                        break
            ind.append(i)
    else:
        ind=list(np.where(index_ins == 2)[0])

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

def find_first_eff_sol(non_basic_ind, basic_ind, B_inv, N, used_indicies):
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


    
    if index_in==np.inf or index_out==np.inf:
        raise ValueError 
       
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
    #B_inv=np.linalg.pinv(B)
    B_inv = LUsolve(B)


    first_sol = True
    eff = check_efficient(B_inv,CN,CB,N,first_sol,C,b, basic_ind) #Check efficiency of the initial solution

    #Loop to find initial efficient solution

    while not eff:

        basic_ind,non_basic_ind=find_first_eff_sol(non_basic_ind, basic_ind, B_inv, N, used_indicies) #Pivot to try to find first solution
        
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
            #B_inv = np.linalg.pinv(B)
            B_inv = LUsolve(B)
            # print(B)
            # print(B_inv)
        except np.linalg.LinAlgError:
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
            solution_vec.append(B_inv.solve(b))
            eff_ind.append(basic_ind.copy())
        else:
            C_row_sum = -C.sum(axis=0)
            x0 = np.zeros(A.shape[1])

            x0[basic_ind]=B_inv.solve(b)
            print(x0)

            result = linprog(C_row_sum, b_ub = -C@x0, A_ub = -C, A_eq = A, b_eq = b, x0=x0, method="revised simplex")
            print("LINPROG", result)
            # print()
            if np.all(result.x == x0):
                solution_vec.append(B_inv.solve(b))
                eff_ind.append(basic_ind.copy())
                used_indicies.append(basic_ind.copy())

            sol = result.x

            tmp_basic_ind = np.where(np.abs(sol)>1e-6)
            tmp_basic_ind=list(tmp_basic_ind[0])
            # if(len(tmp_basic_ind)>len(basic_ind)):
            #     print("AAAAAAA",tmp_basic_ind)
            if len(tmp_basic_ind)==len(basic_ind):
                print("FOUND SOL", tmp_basic_ind)
                eff = True
                basic_ind=sorted(tmp_basic_ind)

                B = A[:,basic_ind]
                N = A[:,non_basic_ind]
                CN = C[:,non_basic_ind]
                CB = C[:,basic_ind]
                #B_inv=np.linalg.inv(B) 
                B_inv = LUsolve(B)

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
    first_sol = False
    sols=True
    basic_explore = []
    non_basic_explore = []
    while sols:

        basic_ind_list,non_basic_ind_list=find_all_eff_sol(non_basic_ind, basic_ind, B_inv, CN, CB, N, used_indicies)
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
            break
        print("This index will be checked", basic_explore[0])
        # print(basic_explore)
        # print(non_basic_explore)
        # print(used_indicies)

        basic_ind = basic_explore[0]
        non_basic_ind = non_basic_explore[0]
        basic_explore.pop(0)
        non_basic_explore.pop(0)
        
        # print(basic_ind,non_basic_ind)


        B = A[:,basic_ind]
        N = A[:,non_basic_ind]
        CN = C[:,non_basic_ind]
        CB = C[:,basic_ind]
        #B_inv=np.linalg.inv(B) #Can use try for the case that the matrix is singular
        B_inv = LUsolve(B)
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
            solution_vec.append(B_inv.solve(b))
            eff_ind.append(basic_ind.copy())
        else:
            C_row_sum = -C.sum(axis=0)
            x0 = np.zeros(A.shape[1])

            x0[basic_ind]=B_inv.solve(b)

            result = linprog(C_row_sum, b_ub = -C@x0, A_ub = -C, A_eq = A, b_eq = b, x0=x0, method="revised simplex")
            if(basic_ind==[4,8,9]):
                print("LINPROG", result)
            # print()
            sol = result.x
            if np.all(sol == x0):
                print("FOUND SOL", tmp_basic_ind)
                solution_vec.append(B_inv.solve(b))
                eff_ind.append(basic_ind.copy())

            else:

                tmp_basic_ind = np.where(np.abs(sol)>1e-6)
                tmp_basic_ind=list(tmp_basic_ind[0])
                if any(np.array_equal(tmp_basic_ind, explore) for explore in basic_explore):
                        solution_vec.append(B_inv.solve(b))
                        eff_ind.append(tmp_basic_ind.copy())
                        basic_explore.remove(tmp_basic_ind.copy())
                        basic_ind = sorted(tmp_basic_ind)
                        non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]
                        B = A[:,basic_ind]
                        N = A[:,non_basic_ind]
                        CN = C[:,non_basic_ind]
                        CB = C[:,basic_ind]
                        #B_inv=np.linalg.inv(B) 
                        B_inv = LUsolve(B)
                else:

                        
            # if(len(tmp_basic_ind)<len(basic_ind)):
            #     tmp_basic_ind.append()

                    if len(tmp_basic_ind)==len(basic_ind) and not any(np.array_equal(tmp_basic_ind, used) for used in used_indicies):
                        
                        # if any(np.array_equal(tmp_basic_ind, used) for used in used_indicies):
                        #     continue
                        # else:
                        
                        basic_ind = sorted(tmp_basic_ind)
                        non_basic_ind = [x for x in range(num_non_basic+num_basic) if x not in basic_ind]

                        #Add new non-basic list
                        #Also check if the basic list is already explored

                        B = A[:,basic_ind]
                        N = A[:,non_basic_ind]
                        CN = C[:,non_basic_ind]
                        CB = C[:,basic_ind]
                        #B_inv=np.linalg.inv(B) 
                        B_inv=LUsolve(b)

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
print(A)
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

simplex(A,b,C)
