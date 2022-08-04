import teneto
import numpy
from tqdm import tqdm
import tims

nodes = ['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master']
 
loaded = numpy.load("100k_frontend_graphs-temporal.npz")
measures = []
for graph in tqdm(loaded.values()):
    metric = teneto.networkmeasures.temporal_degree_centrality(teneto.TemporalNetwork(from_array=graph, nettype="bu"), calc="overtime")
    if len(metric) == 0:
        continue
    measures.append(metric)
numpy.save("tdc.npy", measures)
print(measures / len(loaded))
