from MOLP_simplex import simplex
from CORT.utils import prob

A,b,C = prob('Liver', True, BDY_downsample=1000, OAR_downsample=50, PTV_downsample=10)

simplex(A,b,C,std_form=True)
