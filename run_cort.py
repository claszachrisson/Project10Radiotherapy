from MOLP_simplex import simplex
import CORT.utils as utils

A,b,C, i = utils.prob('Liver', True, BDY_downsample=10000, OAR_downsample=500, PTV_downsample=100)

indices, result = simplex(A,b,-C,std_form=True, Initial_basic=i)

print(len(indices))

