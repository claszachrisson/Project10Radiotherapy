from MOLP_simplex_binary_used import simplex
import CORT.utils as utils

A,b,C, i = utils.prob('Liver', True, BDY_downsample=10000, OAR_downsample=1000, PTV_downsample=100)

indices, result = simplex(A,b,-C,std_form=True, Initial_basic=i)

print(len(indices))

