from scipy import linalg
import numpy as np

class LUsolve:
    def __init__(self, B):
        self.B = np.array(B)
        self.size = self.B.shape[0]
        self.LU, self.piv = linalg.lu_factor(self.B)
        self.L = np.tril(self.LU,-1) + np.eye(self.size)
        self.U = np.triu(self.LU)
        self.ind = np.arange(self.size)
        self.piv2ind()

    def solve(self,b):
        return linalg.lu_solve((self.LU,self.piv),b)
    
    def piv2ind(self):
        self.ind = np.arange(self.size)
        for i in range(self.size):
            self.ind[i], self.ind[self.piv[i]] = self.ind[self.piv[i]], self.ind[i]
    
    def update(self,j,a):
        new = linalg.solve_triangular(self.L,a, lower=True)
        #print(new)
        old = self.U[:,j].reshape(-1,1)
        #print(old)
        e = np.zeros((self.size,))[np.newaxis]
        e[0,j] = 1
        Ux = self.U + (new-old)@e
        return Ux

    def solve2(self,b,E):
        b_perm = np.array(b)[self.ind]
        y = linalg.solve_triangular(self.L,b_perm, lower=True)
        z = E@y
        return linalg.solve_triangular(self.U,z)
    
    def solve3(self,b):
        self.LU=self.L+self.U-np.eye(self.size)
        return self.solve(b)


B = np.array([[2,0,4,0,-2], [3,1,0,1,0], [-1,0,-1,0,-2],[0,-1,0,0,-6],[0,0,1,0,4]])
Bn = np.array([[2,0,7,0,-2], [3,1,-2,1,0], [-1,0,0,0,-2],[0,-1,3,0,-6],[0,0,0,0,4]])
a = np.array([[1],[0],[3],[3],[7]])
b = [5,0,0,0,-1]

x = LUsolve(B)
z = LUsolve(Bn)
newU = x.update(2,a)
newU[:,[2,3,4]] = newU[:,[3,4,2]]
newU[[2,3,4],:] = newU[[3,4,2],:]
print(newU)
#print(newU)
E = np.eye(x.size)
E[4,2] = -(newU[4,2])/newU[2,2]
#print(newU[4,3], newU[3,3])
E[4,3] = -(newU[4,3]+E[4,2]*newU[2,3])/newU[3,3]
#E[4,2] = -newU[4,2]/newU[3,2]
print(E)
newU = E@newU
print(np.round(newU,2))
#print(np.round(newU,10))
x.U=newU
x.piv = np.array([1,3,3,4,4])
x.piv2ind()


print(z.solve(b))
print(x.solve2(b,E))
#print(x.L)
#print(z.L)
#print(x.U)
#print(z.U)
#print(z.piv)
#b = [1,3,0,2,0]
#b = np.array(b)[x.ind]
#start = time.time()
#for i in range(1000000):
#    y = x.solve2(b)
#print(time.time()-start)
#y = x.solve(b)
#print(y)
#y = x.solve2(b)
#print(y)