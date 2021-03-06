#Example Code for SPOT: 

import numpy as np
import matplotlib.pyplot as plt
from spot import bidSPOT, dSPOT,  SPOT

import pandas as pd
f = './loadtest.csv'
P = pd.read_csv(f)
# stream
u_data = (P['Time'] == '2/17/2022')| (P['Time'] == '3/2/2022')
data = P['orders-db'][u_data].values
# initial batch
u_init_data = (P['Time'] == '10/14/2021') | (P['Time'] == '10/20/2021')| (P['Time'] == '10/19/2021')
init_data = P['orders-db'][u_init_data].values
q = 1e-5  			# risk parameter
s = SPOT(q)  		# SPOT object
s.fit(init_data,data) 	# data import
s.initialize() 		# initialization step
results = s.run() 	# run
s.plot(results) 	# plot


-----------------------------------------------------
#Example Code for bidSPOT: 
import numpy as np
import matplotlib.pyplot as plt
from spot import bidSPOT, dSPOT,  SPOT
import pandas as pd
f = './loadtest.csv'
P = pd.read_csv(f)
# stream
u_data = (P['Time'] == '2/17/2022')| (P['Time'] == '3/2/2022')
data = P['orders-db'][u_data].values
# initial batch
u_init_data = (P['Time'] == '10/14/2021') | (P['Time'] == '10/20/2021')| (P['Time'] == '10/19/2021')
init_data = P['orders-db'][u_init_data].values

q = 1e-5 # risk parameter
d = 10	# depth
s = bidSPOT(q,d)  		# bidSPOT object
s.fit(init_data,data) 	# data import
s.initialize() 		# initialization step
results = s.run() 	# run
s.plot(results) 	# plot
