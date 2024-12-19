import numpy as np
import CORT.utils as utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A,b,C, i = utils.prob('Liver', True, BDY_downsample=10000, OAR_downsample=1000, PTV_downsample=100)

result = np.load(f'result_liver_BDY_downsample_{10000}_OAR_downsample_{1000}_PTV_downsample_{100}.npz')

eff_ind = result['list_data']  
solution_vec = result['array_data']

cost = C @ solution_vec.T

for i in range(cost.shape[0]):
    is_strictly_larger = True
    for j in range(cost.shape[0]):
        if i != j and not np.all(cost[i] < cost[j]):
            is_strictly_larger = False
            break
    if is_strictly_larger:
        print("NOO")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cost[0], cost[1], cost[2], color='r', label='cost point Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
