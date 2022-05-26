#Test Example 01
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
f = './loadtest.csv'
P = pd.read_csv(f)
#perform Granger-Causality test
grangercausalitytests(P[['carts', 'front-end']], 4) # log value=1-4 


---------------------------------------------------------------------

#Test Example 02 : For testing Reverse Correlation 
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
f = './loadtest.csv'
P = pd.read_csv(f)
#perform Granger-Causality test
grangercausalitytests(P[['front-end', 'carts']], 4) # log value=1-4 
