import numpy as np
import CORT.utils as utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A,b,C, i = utils.prob('Liver', True, BDY_downsample=10000, OAR_downsample=1000, PTV_downsample=100)

result = np.load('result_liver_BDY_downsample_{10000}_OAR_downsample_{1000}_PTV_downsample{100}.npz')

eff_ind = result['list_data']  
solution_vec = result['array_data']

cost = C @ solution_vec.T



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cost[0], cost[1], cost[2], color='r', label='cost function values')
ax.set_xlabel('BDY')
ax.set_ylabel('OAR')
ax.set_zlabel('PTV')
ax.legend()
plt.show()
