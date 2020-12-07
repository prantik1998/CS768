import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import csv
import config

from Simulation_utilities import get_gamma_distribution_params,edges2graph,generate_power_law_degrees,random_subset
from Simulation_Results import SimulationResults
from Simulation_steps import infection_step,quarantine_step,incubation_step,recovery_step,testing_step
from Simulation_tests import test_group,tests_analyzed
from Simulation_graphs import generate_scale_free_graph

mpl.rcParams['figure.figsize'] = [25, 10]

def init_states(N,initial_infected_num,incubation_duration_mean,incubation_duration_std,prob_recover,prob_asymptomatic):
	"""Init the state variables of the simulation.

	Each group has an array of 0/1 indicators, and potentially days left array.
	Susceptible (S_arr), Exposed (E_arr, E_left), Infected (I_arr), Asymptomatic (A_arr),
	Recovered (R_arr), Quarantined (Q_arr, Q_left), Tested Positive (TP_arr),
	Time until test results return (T_result_left), what the test results wil be
	when they return (T_result_positive_arr).
	i-th indicator value of 1 means the i-th node is a member of that group.
	*_left are arrays of number of days left until transition to next state. A
	value of -1 or less means Not Available. For example an individual not in the
	E group (i.e. has E_arr[i]==0) does not have a meaningful E_left[i], so it
	will be -1 or less.

	Args:
		prob_recover: per day. Inverse of expected duration of being Infected.
		prob_asymptomatic: Share of population who would not develop symptoms even
			when sick. They are not detected along with the symptomatic Infected in
			the testing step, but are detected by direct tests (mass testing or when
			neighbors are tested).
	"""

	all_nodes_arr = np.arange(N)
	S_arr = np.ones_like(all_nodes_arr)

	# Use expected ratio of exposed to infected.
	initial_exposed_num = int(initial_infected_num * incubation_duration_mean * prob_recover)
	initial_infected_num = int(initial_infected_num)
	# Draw E & I, mutually exclusive.

	exposed_and_infected_inds = np.random.choice(all_nodes_arr, initial_exposed_num + initial_infected_num, replace=False)
	exposed_inds = exposed_and_infected_inds[:initial_exposed_num]
	infected_inds = exposed_and_infected_inds[initial_exposed_num:]
	E_arr = np.zeros_like(S_arr)
	E_arr[exposed_inds] = 1
	I_arr = np.zeros_like(S_arr)
	I_arr[infected_inds] = 1
	A_arr = random_subset(np.ones_like(S_arr), prob_asymptomatic)
	S_arr -= (I_arr + E_arr)
	R_arr = np.zeros_like(S_arr)
	Q_arr = np.zeros_like(S_arr)
	# Time left to exit quarantine.
	Q_left = np.full_like(S_arr, -1, dtype='float')
	# Time left to become infected.
	E_left = np.full_like(S_arr, -1, dtype='float')
	# Seed with incubation periods.
	incubation_k, incubation_theta = get_gamma_distribution_params(incubation_duration_mean, incubation_duration_std)
	incubation_durations = np.random.gamma(incubation_k, incubation_theta, N)
	np.copyto(E_left, incubation_durations, where=(E_arr > 0), casting='safe')
	TP_arr = np.zeros_like(S_arr)
	# Time left until test results come back.
	T_result_left = np.full_like(S_arr, -1)
	# What the test results will be, when they come back.
	T_result_positive_arr = np.zeros_like(S_arr)
	return S_arr, E_arr, E_left, I_arr, A_arr, R_arr, Q_arr, Q_left, TP_arr, T_result_left, T_result_positive_arr

def create_counter(I_arr, E_arr, R_arr, Q_arr, S_arr, n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested):
	"""Counts number of individuals in each group.

	Returns a dict from group name to sum of its indicator array.
	"""
	
	return dict(infected=I_arr.sum(),exposed=E_arr.sum(),recovered=R_arr.sum(),quarantined=Q_arr.sum(),susceptible=S_arr.sum(),n_infected_tested=n_infected_tested,n_neighbors_traced=n_neighbors_traced,n_neighbors_tested=n_neighbors_tested,n_general_tested=n_general_tested)


def simulation(G=None,verbose=True):
	# Process input.
	if G is None:
		if verbose:
			print('Generating graph...')
		G = generate_scale_free_graph(config.N, min_degree=config.MIN_DEGREE, mean_degree=config.MEAN_DEGREE, gamma=config.GAMMA)
		if verbose:
			print('Done!')
	else:
		N = G.number_of_nodes()
	adj_mat = nx.adjacency_matrix(G)

	initial_infected_num = int(config.INITIAL_INFECTED_NUM)	# Make sure it's an integer.
	incubation_k, incubation_theta = get_gamma_distribution_params(config.INCUBATION_DURATION_MEAN,config.INCUBATION_DURATION_STD)

	(S_arr, E_arr, E_left, I_arr, A_arr, R_arr, Q_arr, Q_left, TP_arr,T_result_left, T_result_positive_arr) = init_states(config.N,config.INITIAL_INFECTED_NUM,config.INCUBATION_DURATION_MEAN,config.INCUBATION_DURATION_STD,config.PROB_RECOVER,config.PROB_ASYMPTOMATIC)
	n_infected_tested = n_neighbors_traced = n_neighbors_tested = n_general_tested = 0

	# Step-wise probabilities.
	prob_infect_per_step = config.PROB_INFECT / config.STEPS_PER_DAY
	prob_infected_detected_per_step = config.PROB_INFECTED_DETECTED / config.STEPS_PER_DAY
	prob_exposed_detected_per_step = config.PROB_EXPOSED_DETECTED / config.STEPS_PER_DAY
	prob_recover_per_step = config.PROB_RECOVER / config.STEPS_PER_DAY

	max_step_wise_prob = max(prob_infect_per_step, prob_infected_detected_per_step,prob_exposed_detected_per_step, prob_recover_per_step)
	
	if max_step_wise_prob > 0.5:
		print('WARNING: steps_per_day too small? Maximal step-wise probability =', max_step_wise_prob)

	counters = []

	# Main loop.
	for step_num in range(config.MAX_STEPS):

		# Save metrics.
		counters.append(create_counter(I_arr, E_arr, R_arr, Q_arr, S_arr,n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested))

		# Infection step. S -> E
		S_arr, E_arr = infection_step(S_arr, E_arr, E_left, I_arr, A_arr, Q_arr, adj_mat,incubation_k=incubation_k,incubation_theta=incubation_theta,prob_infect=prob_infect_per_step,prob_infect_exposed_factor=config.PROB_INFECT_EXPOSED_FACTOR,relative_infectiousness_asymptomatic=config.RELATIVE_INFECTIOUSNESS_ASYMPTOMATIC,duration_exposed_infects=config.DURATION_EXPOSED_INFECTS)

		# Incubation step. E -> I
		E_arr, E_left, I_arr = incubation_step(E_arr, E_left, I_arr, config.STEPS_PER_DAY)

		# Testing step. Update tested_positive and newly quarantined.
		(TP_arr, new_Q_arr, T_result_left, T_result_positive_arr, n_infected_tested,n_neighbors_traced, n_neighbors_tested, n_general_tested) = testing_step(E_arr, I_arr, A_arr, TP_arr, T_result_left, T_result_positive_arr,prob_infected_detected_per_step,config.PROB_NEIGHBOR_TRACED,prob_exposed_detected_per_step,config.QUARANTINE_NEIGHBORS,config.TEST_NEIGHBORS,config.TEST_DELAY_TIME,adj_mat,config.STEPS_PER_DAY)

		# Quarantine step. Update Q
		Q_arr, Q_left = quarantine_step(Q_arr, Q_left, R_arr, TP_arr, new_Q_arr,config.DAYS_IN_QUARANTINE,config.STEPS_PER_DAY)

		# Recovery step. I -> R
		I_arr, R_arr = recovery_step(I_arr, R_arr, prob_recover_per_step)

		# Stop condition. No more Infected or Exposed.
		if I_arr.sum() == 0 and E_arr.sum() == 0:
			break
	else:
		print('WARNING: max number of steps reached.')

	# Add final counters.
	counters.append(create_counter(I_arr, E_arr, R_arr, Q_arr, S_arr,n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested))

	# Convert to DataFrame.
	results_df = pd.DataFrame(counters)
	tested = results_df.filter(regex='_tested').sum(axis=1)

	# Test rate: number of tests per day. Smoothe using day-length sliding window.
	results_df['test_rate'] = np.convolve(tested, np.ones((config.STEPS_PER_DAY,)), mode='same')
	results_df['step'] = np.arange(len(results_df))
	results_df['day'] = results_df['step'] / config.STEPS_PER_DAY

	print(f"Gamma :- {config.GAMMA}")

	results_df.to_csv('out_gamma_0.csv',index=False,header=True) 

	results = SimulationResults(results_df, G,prob_infect=config.PROB_INFECT,prob_infected_detected=config.PROB_INFECTED_DETECTED,prob_neighbor_traced=config.PROB_NEIGHBOR_TRACED,prob_exposed_detected=config.PROB_EXPOSED_DETECTED,quarantine_neighbors=config.QUARANTINE_NEIGHBORS,test_neighbors=config.TEST_NEIGHBORS,steps_per_day=config.STEPS_PER_DAY)
	
	if verbose:
		print('Simulation finished!')


	return results

if __name__ == "__main__":
	from config import *
	results = simulation()
	# print(ax_arr[0])


