import numpy as np 

from Simulation_utilities import generate_power_law_degrees,edges2graph

def generate_scale_free_graph(N, min_degree, mean_degree, gamma):
	"""Generates a graph with power-law degree distribution.

	- Draws random degrees according to a power-law distribution.
	- Creates one-sided edges for each node according to its degree.
	- Randomly matches these one-sided edges to another one-sided edge to create
		a single edge.
	- Genereates a graph from this edge list.

	Higher gamma means a fatter tail of the degree distribution.
	Name of returned graph is 'power_law_{gamma}_{min_degree}_{mean_degree}'
	"""
	degrees = generate_power_law_degrees(N, min_degree, mean_degree, gamma)
	# pd.Series(degrees).value_counts().sort_index().to_frame().plot(loglog=True)
	nodes_multiple = np.concatenate([np.full(degree, i) for i, degree in enumerate(degrees)])
	np.random.shuffle(nodes_multiple)
	if nodes_multiple.shape[0] % 2 == 1:
		nodes_multiple = nodes_multiple[:-1]
	edges = nodes_multiple.reshape((nodes_multiple.shape[0] // 2, 2))
	H = edges2graph(edges, N)
	# In Graph name, include the parameters.
	H.name = f'power_law_{round(gamma, 3)}_{min_degree}_{mean_degree}'
	return H


def genereate_local_edges(N, n_divisions, mean_degree):
	"""Generates edges with high clustering coefficient and constant degree.

	Smaller n_divisions -> higher clustering.
	"""
	all_nodes = np.arange(N)
	division_sizes = np.diff(np.linspace(0, mean_degree/2, n_divisions+1).astype(int))
	edges_groups = []
	for division_size in division_sizes:
		np.random.shuffle(all_nodes)
		for i in range(1, division_size + 1):
			# In the random ordering, create edges between each node to the one
			# that is `i` places later.
			edges_groups.append(np.stack([all_nodes[i:], all_nodes[:-i]], axis=1))

	edges = np.concatenate(edges_groups)
	return edges


def generate_local_graph(N, n_divisions, mean_degree):
	"""Generates a graph with a much higher clustering coefficient.

	Smaller n_divisions -> higher clustering.
	"""
	edges = genereate_local_edges(N, n_divisions, mean_degree)

	H = edges2graph(edges, N)
	# In Graph name, include the parameters.
	H.name = f'local_graph_{n_divisions}_{mean_degree}'
	return H


def genereate_local_edges_from_degrees(degrees, n_divisions=4):
	"""Generates edges with high clustering coefficient, using list of degrees.

	Doesn't exactly preserve this list of degrees. It becomes narrower.

	Smaller n_divisions -> higher clustering: clustering coefficient ~ 1/n_divisions
	"""
	N = len(degrees)
	all_nodes = np.arange(N)
	divisions_per_node = [np.diff(np.linspace(0, degree, n_divisions+1).astype(int))
												for degree in degrees]
	edges_groups = []
	for division_ind in range(n_divisions):
		np.random.shuffle(all_nodes)
		for j in range(N):
			node = all_nodes[j]
			n_neighbors = divisions_per_node[node][division_ind]
			# In the random ordering, create edges between each node to the ones
			# that are immediately after it.
			neighbors = all_nodes[j + 1:j + 1 + n_neighbors]
			node_edges = np.stack([np.full_like(neighbors, node), neighbors], axis=1)
			edges_groups.append(node_edges)

	edges = np.concatenate(edges_groups)
	# Drop half of edges, to make the average degree right.
	edges = edges[np.random.random(len(edges)) < 0.5]
	return edges


def generate_local_scale_free_graph(N, n_divisions, min_degree,mean_degree, gamma):
	"""Generates a scale-free graph with a much higher clustering coefficient.

	WARNING: Resulting graph distribution is not the same as generate_scale_free_graph
		with the sama parameters. This one has a thinner tail.
	Smaller n_divisions -> higher clustering: clustering coefficient ~ 1/n_divisions
	Graph name is 'local_power_law_{n_divisions}_{gamma}_{min_degree}_{mean_degree}'
	"""
	degrees = generate_power_law_degrees(N, min_degree, mean_degree, gamma)
	edges = genereate_local_edges_from_degrees(degrees, n_divisions=n_divisions)

	H = edges2graph(edges, N)
	# In Graph name, include the parameters.
	H.name = f'local_power_law_{n_divisions}_{round(gamma, 3)}_{min_degree}_{mean_degree}'
	return H
