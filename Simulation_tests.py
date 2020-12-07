import config 
import numpy as np 

from Simulation_utilities import random_subset

def tests_analyzed(T_result_left, T_result_positive_arr, steps_per_day):
	"""Time left for results to come back decreased, some results come back.

	Args:
		T_result_left: Array. For each agent, number days left until test result comes back.
		T_result_positive_arr: Whether test results (When they come back) will be positive.

	Returns:
		new_TP_arr: New positives.
		T_result_left: Updated, with less time left.
	"""
	# Decrease time left for test results to come back.
	T_result_left = T_result_left - 1. / config.STEPS_PER_DAY
	# Test results come back for some agents.
	test_comes_back_arr = ((T_result_left < config.EPSILON) & (T_result_left > -1)).astype(int)
	new_TP_arr = test_comes_back_arr & T_result_positive_arr
	# No longer waiting for test results.
	T_result_left = np.where(test_comes_back_arr, -1, T_result_left)
	return new_TP_arr, T_result_left


def test_group(tested_arr, carrier_arr, T_result_left, T_result_positive_arr, test_delay_time):
	"""Tests a group, returning test results, results time, number tested.

	Args:
		tested_arr: Indicator array for those to be tested.
		carrier_arr: Of tested, those who will test positive. Only its logical and
			with tested_arr is used.
		T_result_left: Array. For each agent, number days left until test result comes back.
		T_result_positive_arr: Whether test results (When they come back) will be positive.
		test_delay_time: Delay between the time a test is performed and the time
			results return and contact-traced neighbors are tested/quarantined. Days.
	"""
	# Record positive results for the positives (when they come back)

	T_result_positive_arr = T_result_positive_arr | (tested_arr & carrier_arr)
	# Start the clock on new tests being analyzed.
	T_result_left = np.where(tested_arr, test_delay_time, T_result_left)
	# Assumes 0 negatives (false & true). Divide by P(test=positive) for more realistic estimate.
	n_tested = tested_arr.sum()

	return T_result_left, T_result_positive_arr, n_tested



def symptomatics_tested(I_arr, A_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr, prob_infected_detected, test_delay_time):
	"""Some fraction of symptomatics is tested. Asymptomatic patients aren't tested.

	Args:
		I_arr: Indicator array of Infected.
		A_arr: Asymptomatics. Those who would not develop symptoms even if infected.
		TP_arr: Those who tested positive previously. They aren't retested.
		new_TP_arr: Those whose positive test results came back in this step.
		T_result_left: Array. For each agent, number days left until test result comes back.
		T_result_positive_arr: Whether test results (When they come back) will be positive.
		prob_infected_detected: probability *per step*, not per day.
		test_delay_time: Delay between the time a test is performed and the time
			results return and contact-traced neighbors are tested/quarantined. Days.

	Returns:
		T_result_left, T_result_positive_arr, n_infected_tested.
	"""
	# Infected group is tested with some probability. Except Asymptomatic.
	new_symptomatic_tested_arr = random_subset(I_arr & (1 - A_arr), prob_infected_detected)
	# Known positives and those pending test results aren't retested.
	pending_test_results_arr = (T_result_left > 0).astype(int)
	new_symptomatic_tested_arr &= (1 - (TP_arr | new_TP_arr | pending_test_results_arr))

	T_result_left, T_result_positive_arr, n_infected_tested = test_group(new_symptomatic_tested_arr, I_arr, T_result_left, T_result_positive_arr, test_delay_time)

	return T_result_left, T_result_positive_arr, n_infected_tested, new_symptomatic_tested_arr


def general_population_tested(E_arr, I_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr, prob_exposed_detected, test_delay_time):
	"""Some fraction of the general population is tested.

	Args:
		E_arr: Indicator array of Exposed.
		I_arr: Indicator array of Infected.
		TP_arr: Those who tested positive previously. They aren't retested.
		new_TP_arr: Those whose positive test results came back in this step.
		T_result_left: Array. For each agent, number days left until test result comes back.
		T_result_positive_arr: Whether test results (When they come back) will be positive.
		prob_exposed_detected: Share of general population which is tested, *per step*.
		test_delay_time: Delay between the time a test is performed and the time
			results return and contact-traced neighbors are tested/quarantined. Days.

	Returns:
		T_result_left, T_result_positive_arr, n_infected_tested.
	"""
	# Known positives and those pending test results aren't retested.
	pending_test_results_arr = (T_result_left > 0).astype(int)
	# Random subset of entire population tested,
	# except known positives and those pending test results.
	new_genpop_tested_arr = random_subset(1 - (TP_arr | new_TP_arr | pending_test_results_arr),
																				prob_exposed_detected)

	# Carriers (Exposed or Infected) will have positive test results (when back).
	carrier_arr = E_arr | I_arr
	T_result_left, T_result_positive_arr, n_general_tested = test_group(
			new_genpop_tested_arr, carrier_arr, T_result_left, T_result_positive_arr, test_delay_time)

	return T_result_left, T_result_positive_arr, n_general_tested

