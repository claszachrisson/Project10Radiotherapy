from MOLP_simplex_linprog_LU2 import simplex
import CORT.utils as utils
import numpy as np
import matplotlib.pyplot as plt

#Give your specifications for your run
case = 'Prostate'
BDY_downsample = 6000
OAR_downsample = 200
PTV_downsample = 50
plot_cost_function = True

#Process matrices
A,b,C,i = utils.prob(case, BDY_downsample=BDY_downsample, OAR_downsample=OAR_downsample, PTV_downsample=PTV_downsample)

#Run MOLP simplex
indices, result = simplex(A,b,-C,std_form=True, Initial_basic=i, num_sol = 100)

#Save the result to a file (already done by simplex() )
np.savez(f'result_{case}_BDY_downsample_{BDY_downsample}_OAR_downsample_{OAR_downsample}_PTV_downsample_{PTV_downsample}.npz', list_data=indices, array_data=result)

#Check that no solution is strictly better than another one
cost = C @ result.T
for i in range(cost.shape[0]):
    is_strictly_larger = True
    for j in range(cost.shape[0]):
        if i != j and not np.all(cost[i] < cost[j]):
            is_strictly_larger = False
            break
    if is_strictly_larger:
        print(f"Solution with index {i} is strictly better than the solution with index {j}")

if plot_cost_function:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(cost[0], cost[1], cost[2], color='r', label='Cost function values')
    ax.set_xlabel('BDY')
    ax.set_ylabel('OAR')
    ax.set_zlabel('PTV')
    ax.legend()
    plt.show()

