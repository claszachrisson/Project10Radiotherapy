from MOLP_simplex_sparse import simplex
from MOLP_radther import prob

A,b,C = prob('Liver', True)

simplex(A,b,C,std_form=False)
