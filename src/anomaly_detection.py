from spot import SPOT
import traceback

def detect_anomalies(df, graph, anomaly_dates, calibration_dates):
    """Detects anomalies in nodes with the help of time series data (the dataframe)
    for the nodes. Adds an anomaly/no anomaly marker to the nodes that appear in the time series data.
    Graph modification is in-place."""

    # Prepare data
    all_anomaly_data = df[df['Date'].isin(anomaly_dates)]
    # initial batch
    all_calibration_data = df[df['Date'].isin(calibration_dates)]

    csv_nodes = df.columns.drop(['Date', 'Time', 'Unnamed: 0'])
    print("Node names from data csv: ", sorted(csv_nodes))
    graph_nodes = [node[1]['id'] for node in graph.nodes.data()]
    print("Node names from graph: ", sorted(graph_nodes))
    print("Nodes in graph but not in data csv: ", sorted(set(graph_nodes) - set(csv_nodes)))
    print("Nodes in data csv but not in graph: ", sorted(set(csv_nodes) - set(graph_nodes)))
    for node_name in csv_nodes:
        calibration_data_for_node = all_calibration_data[node_name].to_numpy()
        anomaly_data_for_node = all_anomaly_data[node_name].to_numpy()

        try:
            q = 1e-5  			# risk parameter
            s = SPOT(q)  		# SPOT object
            s.fit(calibration_data_for_node, anomaly_data_for_node) 	# data import
            s.initialize(verbose=False) 		# initialization step
            results = s.run() 	# run
            alarms = results["alarms"]
            node = [n for n in graph.nodes.data() if n[1]["id"] == node_name][0]
            if len(alarms) > 0:
                node[1]["anomaly"] = True
                anomaly_timestamps = [" ".join(all_anomaly_data.iloc[alarm][['Date', 'Time']]) for alarm in alarms]
                node[1]["anomaly_timestamps"] = anomaly_timestamps
            else:
                node[1]["anomaly"] = False
        except Exception:
                print(traceback.format_exc())
