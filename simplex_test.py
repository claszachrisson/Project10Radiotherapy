import numpy as np
# pivot_column = A[:, s]

# # Calculate the ratios b_i / a_i^s where a_i^s > 0
# ratios = b / pivot_column

# # Only consider rows where pivot_column > 0 (i.e., a_i^s > 0)
# positive_ratios = ratios[pivot_column > 0]

# # Calculate the minimum ratio
# t_s = np.min(positive_ratios)

def check_efficient(B_inv,CN,CB,b,N):
    
    CN_eff = CN-CB@B_inv@N
    N_eff = B_inv@N
    Cx = CB@B_inv@b

    pos_nonzero = np.any(CN_eff < 0 | np.all(CN_eff==0), axis=0) #Check if any column is all negative or if all zero (check correctness)
    # print(pos_nonzero)
    # print(CN_eff)
    if(np.any(pos_nonzero==False)):
        return False

    ideal_sol = np.all(CN_eff<=0)
    if(ideal_sol==True):
        print("Ideal solution")
        return True
    
    any_row_negative = np.all(CN_eff < 0, axis=1)
    if(np.any(any_row_negative==True)):
        print("Any row negative")
        return True



    return False

def move_adjacent(non_basic_ind, basic_ind, B, B_inv,N):
    # index=9999 #remove
    As = np.copy(N)
    # print(B_inv)
    # print(N)x
    As = B_inv@N
    b_eff = B_inv@b
    # print(b_eff)

    # As[2,0]=-1
    for i in range(len(non_basic_ind)):
        neg_col = np.all(As[:, i] < 0) #Check if column is negative (Not correct)
        if(neg_col):
            pivot_ind = non_basic_ind[i]
            #Here add how to switch
            break
    #H

    # print(As)
    if(neg_col==False):
        min_value = np.inf
        index_in, index_out = np.inf, np.inf
        for i in range(len(b_eff)):
            for s in range(len(As[0,:])):
                if As[i,s]>=0:
                    value=b_eff[i]/As[i,s]
                else:
                    value = np.inf
                # print(value)
                
                if(value<min_value):
                    print(As[i,s])
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
    print(min_value,(index_in,index_out))
    print(basic_ind,non_basic_ind)


    

    # print(As)
    # print(neg_col)
    # indices = feasible.nonzero()
    # print(indices)
    # if(A)
    # bi=B[:,basic_ind]
    
    # ts=np.min()
    # print(As)
    return basic_ind,non_basic_ind

def simplex(A,b,C):
    # L1 = 0
    # L2 = 0
    #B är Alla basic columner av A
    #Kolla om efficient:
    #b_streck = B^-1*b
    #C_streckN = CN-CB*B^-1*N

    #CB*b = Cx med streck
    #Cx utan streck = CB*B^-1*b +(CN-CB*B^-1*N)*xN
    elems = len(C[0,:])
    num_basic = len(b)
    num_non_basic = len(C[0,:])-num_basic
    # print(num_non_basic)
    # print(num_basic)
 
    basic_ind = np.arange(num_non_basic, num_basic + num_non_basic) #Initial feasible solution
    non_basic_ind = np.arange(0,num_non_basic)
    # print(basic_ind,non_basic_ind)
    # print(A)
    B = A[:,basic_ind]
    N = A[:,non_basic_ind]
    CN = C[:,non_basic_ind]
    CB = C[:,basic_ind]
    # print(B)
    

    # test_arr = np.array([[-1,-2,0],[-1,0,2],[1,0,-1]])

    # pos_nonzero = np.any(test_arr <= 0 | test_arr[i]==0, axis=0)
    # print(pos_nonzero)
    for i in range(10):

        B_inv=np.linalg.inv(B)

        eff = check_efficient(B_inv,CN,CB,b,N)
        if(eff):
            print("EFFICIENT")
            print(basic_ind,non_basic_ind)
            return
        # print(eff)
        # print(non_basic_ind)
        basic_ind,non_basic_ind=move_adjacent(non_basic_ind, basic_ind, B, B_inv, N)

        B = A[:,basic_ind]
        N = A[:,non_basic_ind]
        CN = C[:,non_basic_ind]
        CB = C[:,basic_ind]

    Cx = CB@B_inv@b

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

A = np.array([[1,1,2],[1,2,1],[2,1,1]])
A = np.hstack((A, np.eye(A.shape[0])))
# print(A)
b = np.array([12,12,12])
C = np.array([[6,4,5],[0,0,1]])
C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))

simplex(A,b,C)
