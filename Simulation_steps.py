import config
import numpy as np

from Simulation_utilities import random_subset,individual_infection_probabilities,spread_to_neighbors,contact_tracing
from Simulation_tests import tests_analyzed,symptomatics_tested,general_population_tested



def incubation_step(E_arr, E_left, I_arr, steps_per_day):
	"""E -> I. Incubation period ends for some individuals. They become Infected.

	Args:
		E_left: Array of days left in incubation. When reaches 0, become infectious.
	"""
	E_left = E_left - 1 / steps_per_day	# Decrease time left in incubation.
	# Those whose incubation period ends.
	become_infected = E_arr & (E_left < config.EPSILON)
	E_arr = E_arr - become_infected
	I_arr = I_arr + become_infected
	return E_arr, E_left, I_arr


def recovery_step(I_arr, R_arr, prob_recover):
	"""I -> R. Infected individuals Recover with given step-wise probability."""
	new_recovered = (np.random.random(I_arr.shape) < prob_recover) & I_arr
	R_arr = R_arr + new_recovered
	I_arr = I_arr - new_recovered
	return I_arr, R_arr

def infection_step(S_arr, E_arr, E_left, I_arr, A_arr, Q_arr, adj_mat,incubation_k, incubation_theta, prob_infect,prob_infect_exposed_factor,relative_infectiousness_asymptomatic,duration_exposed_infects):
	"""S -> E. Infectious individuals infect their neighbors with some probability.

	Individuals who are infectious are those who are in I but not in Quarantine
	and those who are Exposed (during final infectiousness duration) but not in Quarantine.
	Infectious individuals interact with each of their neighbors once (or multiple
	times if there are multiple edges between them).
	Each interaction with a neighbor infects that neighbor with probability
	prob_infect (for Infected) and prob_infect * prob_infect_exposed_factor (for Exposed),
	but only if that neighbor is Susceptible and not Quarantined.
	For Asymptomatics, their probabilities of infecting (in both Infected and Exposed)
	are multiplied by relative_infectiousness_asymptomatic.

	All infected individuals become Exposed, and are given an incubation time.

	Args:
		prob_infect: Probability of infection *per step* (not per day).
		incubation_k, incubation_theta: parameters of incubation period length
			in *days*, not steps.
		prob_infect_exposed_factor: Ratio of infectiousness between Exposed (in the
			final days before becoming Infected) and Infected. In other words,
			prob_infect_exposed = prob_infect * prob_infect_exposed_factor.
			Set this to 0 to make the Exposed non-infectious.
		relative_infectiousness_asymptomatic: Ratio of infectiousness between
			Asymptomatic and symptomatic. In other words,
			prob_infect_asymptomatic = prob_infect * relative_infectiousness_asymptomatic.
			The same ratio holds for the final contagious part of the incubation period.
			Set this to 0 to make the Asymptomatic non-infectious.
		duration_exposed_infects: How many days before developing symptoms (becoming
			Infected) an Exposed individual is contagious.
	"""
	# TODO: Can unite these 4 matrix multiplications into 1, to optimize.

	# Those who can be infected - Quarantined cannot be.
	susceptible_not_quarantined = S_arr & (1 - Q_arr)

	# Newly infected by symptomatic Infected who are not Quarantined.
	new_infected = spread_to_neighbors(
			I_arr & (1 - A_arr) & (1 - Q_arr), prob_infect, adj_mat, susceptible_not_quarantined)

	# Newly infected by Asymptomatic Infected who are not Quarantined.
	prob_infect_asymptomatic = prob_infect * relative_infectiousness_asymptomatic
	new_infected |= spread_to_neighbors(
			I_arr & A_arr & (1 - Q_arr), prob_infect_asymptomatic, adj_mat, susceptible_not_quarantined)

	# Newly infected by the Exposed who are infectious, and not Quarantined.
	# Only infectious a few days before developing symptoms.
	E_infectious_arr = E_arr & (E_left <= duration_exposed_infects)
	# Exposed are less infectious.
	prob_infect_exposed = prob_infect * prob_infect_exposed_factor
	# Infections by Exposed who will become symptomatic.
	new_infected |= spread_to_neighbors(
			E_infectious_arr & (1 - A_arr) & (1 - Q_arr), prob_infect_exposed,
			adj_mat, susceptible_not_quarantined)
	# Infections by Exposed who will become Asymptomatic.
	prob_infect_exposed_asymptomatic = prob_infect_exposed * relative_infectiousness_asymptomatic
	new_infected |= spread_to_neighbors(
			E_infectious_arr & A_arr & (1 - Q_arr), prob_infect_exposed_asymptomatic,
			adj_mat, susceptible_not_quarantined)

	# new_infected are now Exposed.
	E_arr = E_arr | new_infected
	incubation_durations = np.random.gamma(incubation_k, incubation_theta, S_arr.shape[0])
	np.copyto(E_left, incubation_durations, where=(new_infected > 0), casting='safe')
	S_arr = S_arr & (1 - new_infected)
	return S_arr, E_arr


def quarantine_step(Q_arr, Q_left, R_arr, TP_arr, new_Q_arr,days_in_quarantine,steps_per_day):
	"""Individuals exit quarantine. Those who tested positive enter quarantine.

	Quarantined people whose time is up exit Quarantine.
	Those who tested positive and haven't recovered stay in Quarantine.
	Those who tested positive and have recovered exit Quarantine, even if their time isn't up.
	Those who should (tested positive or traced neighbors) enter Quarantine.
	Quarantined individuals can neither infect nor be infected.

	Args:
		Q_left: Array of days left in quarantine. When reaches 0, quarantine ends.
		TP_arr: array indicating those who tested positive.
		new_Q_arr: array indicating those who should enter Quarantine in this step.
	"""
	# Exiting quarantine
	# ------------------
	Q_left = Q_left - 1. / steps_per_day	# Decrease time left in Quarantine.
	# Known positives who haven't recovered.
	known_positives = TP_arr & (1 - R_arr)
	known_recovered = TP_arr & R_arr
	# Release those whose Quarantine time is up.
	exit_quarantine = Q_arr & (Q_left < config.EPSILON)
	# Don't release known positives who haven't recovered.
	exit_quarantine &= (1 - known_positives)
	# Release known positives who have recovered (even if their 14 days aren't up).
	exit_quarantine |= known_recovered & Q_arr
	# Apply exiting quarantine.
	Q_arr &= (1 - exit_quarantine)
	# Entering quarantine
	# ------------------
	new_in_quarantine = new_Q_arr & (1 - Q_arr)
	Q_arr |= new_in_quarantine
	# Start the clock on time in Quarantine.
	np.copyto(Q_left, days_in_quarantine, where=(new_in_quarantine > 0), casting='safe')
	return Q_arr, Q_left

def testing_step(E_arr, I_arr, A_arr, TP_arr, T_result_left, T_result_positive_arr,prob_infected_detected,prob_neighbor_traced,prob_exposed_detected,quarantine_neighbors,test_neighbors,test_delay_time,steps_per_day,adj_mat):
	"""Testing of subsets of Infected, the general population. Contact tracing.

	A test comes out positive if a node is either Exposed or Infected.
	Test symptomatics: a random subset of Infected people are tested (and found
		positive). Symptomatics are quarantined even before test results return.
	Mass-testing: a random subset of the population is tested (Exposed & Infected
		people test positive.)
	Contact tracing: a random subset of neighbors of those who tested positive,
	are traced, and then either tested or quarantined without being tested.
	Counts number of tests performed, assuming 0 false negative tests.
	Known positives and those pending test results are not retested.

	Args:
		A_arr: Asymptomatics. Those who would not develop symptoms even if infected.
		TP_arr: Those who tested positive previously. They aren't retested.
		T_result_left: Array. For each agent, number days left until test result comes back.
		T_result_positive_arr: Whether test results (When they come back) will be positive.
		prob_infected_detected: probability *per step*, not per day.
		prob_neighbor_traced: absolute probability, not per day / step.
		prob_exposed_detected: Share of general population which is tested, *per step*.
		quarantine_neighbors: When an individual tests positive, whether their
			traced neighbors are quarantined.
		test_neighbors: When an individual tests positive, whether their traced
			neighbors are tested.
		test_delay_time: Delay between the time a test is performed and the time
			results return and contact-traced neighbors are tested/quarantined. Days.
		adj_mat: not required if prob_neighbor_traced=0.

	Returns:
		TP_arr, new_Q_arr, T_result_left, T_result_positive_arr, n_infected_tested,
		n_neighbors_traced, n_neighbors_tested, n_general_tested
	"""
	# TODO: Testing for other populations (high degree nodes).
	new_TP_arr, T_result_left = tests_analyzed(
			T_result_left, T_result_positive_arr, steps_per_day)

	(T_result_left, T_result_positive_arr, n_infected_tested,
	 new_symptomatic_tested_arr) = symptomatics_tested(
			 I_arr, A_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr,
			 prob_infected_detected, test_delay_time)

	T_result_left, T_result_positive_arr, n_general_tested = general_population_tested(
			E_arr, I_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr,
			prob_exposed_detected, test_delay_time)

	# Indicator for those who need to enter Quarantine. New positives and symptomatic.
	new_Q_arr = new_TP_arr | new_symptomatic_tested_arr
	# TODO: Add non-covid symptomatics who are quarantined, and released on negative test results.

	(new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced,
	 n_neighbors_tested) = contact_tracing(E_arr, I_arr, TP_arr, new_TP_arr,
																				 T_result_left, T_result_positive_arr,
																				 new_Q_arr, prob_neighbor_traced,
																				 quarantine_neighbors, test_neighbors,
																				 test_delay_time, adj_mat)
	TP_arr = TP_arr | new_TP_arr
	return TP_arr, new_Q_arr, T_result_left, T_result_positive_arr, n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested
