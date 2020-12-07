import numpy as np
import networkx as nx

def get_gamma_distribution_params(mean, std):
	"""Turn mean and std of Gamma distribution into parameters k and theta."""
	# mean = k * theta
	# var = std**2 = k * theta**2
	theta = std**2 / mean
	k = mean / theta
	return k, theta


def edges2graph(edges, N=10000):
	"""Creates MultiGraph from list of edges."""
	H = nx.MultiGraph()
	H.add_nodes_from(np.arange(N))
	H.add_edges_from(edges)
	return H

def generate_power_law_degrees(N=10000, min_degree=2, mean_degree=20, gamma=0.2):
	"""Generate array of degrees according to a power law distribution.

	If U~U[0,1] then generate power law values as
	X = U^(-gamma) - 1
	and then stretch them linearly to provide mean and min as required.
	Then convert them to integers with probability of being rounded up equal to
	the fractional part of the number.
	"""
	eps = 1e-10
	power_law_values = 1 / np.random.random(N)**gamma - 1 + eps
	value_mean = np.mean(power_law_values)
	frac_degrees = min_degree + power_law_values / value_mean * (mean_degree - min_degree)
	int_degrees = frac_degrees.astype(int) + (np.random.random(N) < (frac_degrees % 1))
	return int_degrees

def random_subset(indicator_arr, sample_probs):
	"""In array of 0,1 leave subset of 1s with probabilities sample_probs.

	Args:
		indicator_arr: An array of 0,1 values.
		sample_probs: A number or an array with same shape as indicator_arr, for
			individual heterogeneous sampling probabilities.
	"""
	subset_arr = (np.random.random(indicator_arr.shape) < sample_probs) & indicator_arr
	return subset_arr

def individual_infection_probabilities(I_arr, prob_infect, adj_mat):
	"""Individual probabilities of being infected by the infectious in I_arr.

	Counts the number of infectious neighbors, and calculates individual
	probability of being infected.
	Probability of being infected boy each connection is prob_infect.

	Args:
		prob_infect: Probability of infecting a single neighbor.
		adj_mat: Adjacency matrix of the graph.
	"""
	if prob_infect == 0:
		# Performance optimization - Don't count edges when no chance of infection.
		return np.zeros(I_arr.shape)

	# Number of connections to infected (could be multiple connections from the same infected)
	connections_to_I = number_of_edges_to_group(I_arr, adj_mat)
	# Individual probability of not being infected: (1 - p_infect)**n_connections
	no_infection_probs = np.exp(np.log(1 - prob_infect) * connections_to_I)
	return 1 - no_infection_probs

def spread_to_neighbors(from_group_arr, prob_spread, adj_mat, to_group=None):
	"""Neighbors of the group are marked with a certain probability for each edge.

	Each member of the group denoted by from_group_arr spreads to each one of its
	neighbors with probability prob_spread. This can be used for infection,
	tracing or anything else.

	Args:
		from_group_arr: Indicator array of seed group, from which the spread emanates.
		prob_spread: Probability that each member of the group "marks" or spreads
			to each of its neighbors.
		adj_mat: Adjacency matrix of the graph.
		to_group: Indicator array. Restrict spread to members of this group. Default
			is no restriction - the entire population.

	Returns:
		Indicator array of 1s for those who were marked.
	"""
	spread_probs = individual_infection_probabilities(from_group_arr, prob_spread, adj_mat)
	if to_group is None:
		to_group = np.ones_like(from_group_arr)
	return random_subset(to_group, spread_probs)

def contact_tracing(E_arr, I_arr, TP_arr, new_TP_arr, T_result_left,T_result_positive_arr, new_Q_arr, prob_neighbor_traced,quarantine_neighbors, test_neighbors, test_delay_time, adj_mat):
	"""Contact tracing. Neighbors of positives are traced and tested / quarantined.

	For each positive, each neighbor is traced with probability prob_neighbor_traced.

	Args:
		quarantine_neighbors: When an individual tests positive, whether their
			traced neighbors are quarantined.
		test_neighbors: When an individual tests positive, whether their traced
			neighbors are tested.

	Returns:
		new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced, n_neighbors_tested.
	"""

	# Check if no neighbors are tested, to avoid matrix multiplication if not required.
	if prob_neighbor_traced == 0:
		n_neighbors_tested = 0
		n_neighbors_traced = 0
		return new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced, n_neighbors_tested

	neighbors_traced = spread_to_neighbors(new_TP_arr, prob_neighbor_traced, adj_mat)
	n_neighbors_traced = neighbors_traced.sum()
	n_neighbors_tested = 0

	if quarantine_neighbors:
		# Quarantine traced neighbors.
		# Known positives aren't newly quarantined (either covered elsewhere or recovered).
		new_Q_arr = new_Q_arr | (neighbors_traced & (1 - (new_TP_arr | TP_arr)))

	if test_neighbors:
		# Test the traced neighbors.
		# Carriers (Exposed or Infected) will have positive test results (when back).
		carrier_arr = E_arr | I_arr
		# Known positives and those pending test results aren't retested.
		pending_test_results_arr = (T_result_left > 0).astype(int)
		neighbors_tested = neighbors_traced & (1 - (TP_arr | new_TP_arr | pending_test_results_arr))
		T_result_left, T_result_positive_arr, n_neighbors_tested = test_group(
				neighbors_tested, carrier_arr, T_result_left, T_result_positive_arr, test_delay_time)

	return new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced, n_neighbors_tested

def number_of_edges_to_group(group_arr, adj_mat):
	"""Calculates number of edges from each node to nodes in group_arr.

	Returns array of integers representing number of edges for each node.
	Multiple edges are counted as their multiplication.

	Args:
		group_arr: Indicator array - 1 indicates group membership.
		adj_mat: Adjacency matrix of the graph.

	Returns:
		Array indicating for each member of the population the number of neighbors
		they have in the group indicated by group_arr.
	"""
	if (group_arr == 0).all():
		# Performance optimization. If the group is empty, don't count edges.
		return np.zeros_like(group_arr)

	# Adjacency matrix filtered down to neighbors of group.
	adj_mat_group = adj_mat.multiply(group_arr)
	# number of connections to infected (could be multiple connections from the same infected)
	connections_to_group = np.array(adj_mat_group.sum(axis=1))[:, 0]

	return connections_to_group