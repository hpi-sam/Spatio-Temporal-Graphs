# Exploring Spatio-Temporal Graphs as Means to Identify Failure Propagation

This repository includes our data and code for exploring failure propagation in spatio-temporal graphs.

Our approach is separated in three subtasks:
- Generating Anomaly Propagation Subgraphs from a system structure and given time series as source subgraphs
- Exploring Synthetic Subgraph Generation loosely based on a given system structure as target subgraphs
- Graph Matching of source and target subgraphs using Graph Auto Encoder 

The code for those subtasks is available in the `/src` folder.

The underlying system graph structure and timeseries data is available in the `/data` folder. This folder also contains the output of the source and target subgraph generation. The output is provided as numpy arrays and stored in .npy files. To decode and encode those graphs for use in networkx graph structures, see `src/correlation.py`.

For calculating the Temporal Centrality Metrics the overtime package was used. The overtime directory was cloned directly from the [overtime3 package]{https://github.com/overtime3/overtime}.

The project seminar slides are available in the `/slides` folder and information on the project scope definition is given in the `/project scope` folder.

For code provided by the instructors, see `/SourceCode` folder.