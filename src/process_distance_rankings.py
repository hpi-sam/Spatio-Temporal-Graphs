# %% [markdown]
# # Experiments for Graph Matching
# This notebook contains the experiments to compare the graph matching results via embeddings produced by a GNN to graph matching results via embeddings that are represented by temporal centralities. 

# %%
import numpy as np
import matplotlib.pyplot as plt 

# %%
temporal_betweenness_vectors_sources = np.load('../data/temporal_betweenness_vectors_overtime.npy')
temporal_closeness_vectors_sources = np.load('../data/temporal_closeness_vectors_overtime.npy')
temporal_degree_vectors_sources = np.load('../data/temporal_degree_vectors_overtime.npy')

# %%
temporal_betweenness_vectors_targets = np.load('artificial-graph-generation/generated/temporal_betweennesses_unique.npy')
temporal_closeness_vectors_targets = np.load('artificial-graph-generation/generated/temporal_closenesses_unique.npy')
temporal_degree_vectors_targets = np.load('artificial-graph-generation/generated/temporal_degrees_unique.npy')

# %%
source_encodings = np.load("../data/anomaly_source_embeddings_17_dims_30_epochs.npy")
target_encodings = np.load("../data/anomaly_target_embeddings_17_dims_30_epochs.npy")

# %%
num_sources = len(temporal_betweenness_vectors_sources)
assert len(temporal_closeness_vectors_sources) == len(temporal_degree_vectors_sources) and len(temporal_closeness_vectors_sources) == num_sources
num_targets = len(temporal_betweenness_vectors_targets)
assert len(temporal_closeness_vectors_targets) == len(temporal_degree_vectors_targets) and len(temporal_closeness_vectors_targets) == num_targets


# %% [markdown]
# ## Comparison of GNN to temporal betweenness

# %%
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.metrics import dcg_score
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import gzip

# %%
#source_indices = np.random.default_rng().choice(num_sources, 100, replace=False)
source_indices = list(range(num_sources))

# %%
# Temporal closeness
temporal_closeness_distances = []
temporal_closeness_rankings = []
for i in tqdm(source_indices, desc="Temporal Closeness"):
    temporal_closeness_distances.append([])
    for j in range(num_targets):
        temporal_closeness_distances[-1].append(distance.euclidean(temporal_closeness_vectors_sources[i], temporal_closeness_vectors_targets[j]))
    argsorted = np.argsort(temporal_closeness_distances[-1])
    result = np.arange(num_targets)
    for index in range(num_targets):
        result[argsorted[index]] = index
    temporal_closeness_rankings.append(result)
temporal_closeness_distances = np.array(temporal_closeness_distances, dtype=np.float32)
temporal_closeness_rankings = np.array(temporal_closeness_rankings, dtype=np.int32)
with gzip.open("temporal_closeness_distances.npy", "wb") as filehandle:
    np.save(filehandle, temporal_closeness_distances)
with gzip.open("temporal_closeness_rankings.npy", "wb") as filehandle:
    np.save(filehandle, temporal_closeness_rankings)

# %%
# Temporal degree
temporal_degree_distances = []
temporal_degree_rankings = []
for i in tqdm(source_indices, desc="Temporal Degree"):
    temporal_degree_distances.append([])
    for j in range(num_targets):
        temporal_degree_distances[-1].append(distance.euclidean(temporal_degree_vectors_sources[i], temporal_degree_vectors_targets[j]))
    argsorted = np.argsort(temporal_degree_distances[-1])
    result = np.arange(num_targets)
    for index in range(num_targets):
        result[argsorted[index]] = index
    temporal_degree_rankings.append(result)
temporal_degree_distances = np.array(temporal_degree_distances, dtype=np.float32)
temporal_degree_rankings = np.array(temporal_degree_rankings, dtype=np.int32)
with gzip.open("temporal_degree_distances.npy", "wb") as filehandle:
    np.save(filehandle, temporal_degree_distances)
with gzip.open("temporal_degree_rankings.npy", "wb") as filehandle:
    np.save(filehandle, temporal_degree_rankings)

# %%
temporal_betweenness_distances = []
temporal_betweenness_rankings = []
for i in tqdm(source_indices, desc="Temporal Betweenness"):
    temporal_betweenness_distances.append([])
    for j in range(num_targets):
        temporal_betweenness_distances[-1].append(distance.euclidean(temporal_betweenness_vectors_sources[i], temporal_betweenness_vectors_targets[j]))
    argsorted = np.argsort(temporal_betweenness_distances[-1])
    result = np.arange(num_targets)
    for index in range(num_targets):
        result[argsorted[index]] = index
    temporal_betweenness_rankings.append(result)
temporal_betweenness_distances = np.array(temporal_betweenness_distances, dtype=np.float32)
temporal_betweenness_rankings = np.array(temporal_betweenness_rankings, dtype=np.int32)
with gzip.open("temporal_betweenness_distances.npy", "wb") as filehandle:
    np.save(filehandle, temporal_betweenness_distances)
with gzip.open("temporal_betweenness_rankings.npy", "wb") as filehandle:
    np.save(filehandle, temporal_betweenness_rankings)

# %%
# Embeddings
embedding_distances = []
embedding_rankings = []
for i in tqdm(source_indices):
    embedding_distances.append([])
    for j in range(num_targets):
        embedding_distances[-1].append(distance.euclidean(source_encodings[i], target_encodings[j]))
    argsorted = np.argsort(embedding_distances[-1])
    result = np.arange(num_targets)
    for index in range(num_targets):
        result[argsorted[index]] = index
    embedding_rankings.append(result)
embedding_distances = np.array(embedding_distances, dtype=np.float32)
embedding_rankings = np.array(embedding_rankings, dtype=np.int32)
with gzip.open("embedding_distances.npy", "wb") as filehandle:
    np.save(filehandle, embedding_distances)
with gzip.open("embedding_rankings.npy", "wb") as filehandle:
    np.save(filehandle, embedding_rankings)
