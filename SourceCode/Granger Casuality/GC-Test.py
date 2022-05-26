#Example
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
f = './loadtest.csv'
P = pd.read_csv(f)
#perform Granger-Causality test
grangercausalitytests(P[['carts', 'front-end']], 4) # log value=1-4 
