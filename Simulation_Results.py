import statsmodels.api as sm
import numpy as np
import config
import matplotlib.pyplot as plt
import pandas as pd

class SimulationResults(object):
	"""Simulation run results: metadata, aggregates and time series."""

	def __init__(self, results_df, G=None, **kwargs):
		"""Initialize with simulation results, graph and hyper-parameters."""
		self.df = results_df
		self.hyperparams = kwargs
		for name, value in kwargs.items():
			setattr(self, name, value)
		if G is None:
			self.N = results_df[['susceptible', 'exposed', 'infected', 'recovered']].iloc[0].sum()
		else:
			self.N = len(G)
			# Extract graph parameters from its name.
			if G.name.startswith('power_law'):
				self.G_attrs = dict(zip(['gamma', 'min_degree', 'mean_degree'], map(float, G.name.split('_')[-3:])))
				self.G_attrs['N'] = self.N

		if not hasattr(self, 'steps_per_day'):
			self.steps_per_day = ((results_df['step'].iloc[1] - results_df['step'].iloc[0]) /
														(results_df['day'].iloc[1] - results_df['day'].iloc[0]))

		self.analyze_results_df()

	def calculate_doubling_time(self):
		"""Returns doubling time of initial spread (in days)."""
		results_df = self.df
		# Exponential regime before leveling off.
		idx_end = (results_df['exposed'] > results_df['exposed'].max() * 0.5).to_numpy().nonzero()[0][0]
		if self.peak_exposed_time < 3 or idx_end == 0:
			# 3 days peak time, or peak value is less than twice initial value - probably containment.
			return np.inf

		# Don't start from the very beginning, to avoid the initial dip in the
		# number of exposed, due to imperfect starting conditions.
		# Start when number of exposed > 2 * minimum (after passing the minimum).
		exposed_min = results_df['exposed'][:idx_end].min()
		idx_min = results_df['exposed'][:idx_end].idxmin()
		start_candidates = ((results_df.index >= idx_min) &
												(results_df.index < idx_end) &
												(results_df['exposed'] > exposed_min * 2)).to_numpy().nonzero()[0]
		if not start_candidates.size:
			# Empty - no candidates. Probably containment.
			return np.inf
		idx_start = start_candidates[0]

		# Linear regression to find doubling time: log2(exposed) ~ day + const
		try:
			X = sm.add_constant(results_df[idx_start:idx_end][['day']], prepend=False)
			log2_exposed = np.log2(results_df[idx_start:idx_end]['exposed'])
			regression_results = sm.OLS(log2_exposed, X).fit()
			# Days for doubling is the inverse of the doubling effect of one day.
			doubling_days = 1 / regression_results.params['day']
		except ValueError:
			doubling_days = None
		return doubling_days

	def calculate_halving_time(self):
		"""Returns halving time of spread after peak (in days)."""
		results_df = self.df
		# Find peak.
		idx_peak = results_df['exposed'].idxmax()
		# Find end point for calculation, not right at the peak, but not at the noisy end.
		end_candidates = ((results_df.index >= idx_peak) &
											(results_df['exposed'] < self.peak_exposed / 5) &
											(results_df['exposed'] > 5)).to_numpy().nonzero()[0]
		if not end_candidates.size:
			# Halving is too short/noisy to calculate the halving time.
			return None

		idx_end = end_candidates[0]
		idx_start = idx_peak
		if idx_end - idx_start < 20:
			# Halving is too short to calculate the halving time.
			return None

		# Linear regression to find halving time: log2(exposed) ~ day + const
		try:
			X = sm.add_constant(results_df[idx_start:idx_end][['day']], prepend=False)
			log2_exposed = np.log2(results_df[idx_start:idx_end]['exposed'])
			regression_results = sm.OLS(log2_exposed, X).fit()
			# Days for halving is the inverse of the halving effect of one day.
			halving_days = -1 / regression_results.params['day']
		except ValueError:
			halving_days = None
		return halving_days

	def analyze_results_df(self):
		"""Calculate various summary stats."""

		results_df = self.df
		self.duration = results_df['day'].iloc[-1]
		# Find peak infections.
		self.peak_infected_time = results_df['day'].iloc[results_df['infected'].idxmax()]
		self.peak_infected = results_df['infected'].max()
		self.peak_fraction_infected = results_df['infected'].max() / self.N
		self.peak_exposed_time = results_df['day'].iloc[results_df['exposed'].idxmax()]
		self.peak_exposed = results_df['exposed'].max()
		self.doubling_days = self.calculate_doubling_time()
		self.halving_days = self.calculate_halving_time()

		# Other result summary stats.
		self.fraction_infected = results_df['recovered'].iloc[-1] / self.N
		# Units: [steps] * [fraction of population]
		fraction_quarantine_steps = results_df['quarantined'].sum() / self.N
		# Units: [days] * [fraction of population]
		self.fraction_quarantine_time = fraction_quarantine_steps / self.steps_per_day
		total_tests = results_df['test_rate'].sum() / self.steps_per_day
		# Number of tests performed, as fraction of the population.
		self.fraction_tests = total_tests / self.N
		self.peak_test_rate = results_df['test_rate'].max() / self.N

	def plot_trends(self, fraction_of_population=True, hyperparams=True, G_attrs=False, columns=None, vertical=True):
		"""Plots the time series of self.df, with the specified columns."""
		
		if columns is None:
			columns = config.MAIN_GROUPS
		title = ''
		if hyperparams:
			title += str({k: round(v, 3) for k, v in self.hyperparams.items()})
		if G_attrs:
			title += str(self.G_attrs)

		if fraction_of_population:
			scale = self.N
			ylabel = 'Fraction of population'
		else:
			scale = 1
			ylabel = 'Individuals'

		if vertical:
			fig, ax_arr = plt.subplots(2, 1, figsize=(10, 20))
			ax_arr[0].set_title('Epidemic Simulation')
			ax_arr[1].set_title('log scale')
		else:
			fig, ax_arr = plt.subplots(1, 2)
		fig.suptitle(title, fontsize=18)
		results_to_plot = self.df.drop('step', axis=1).set_index('day') / scale
		results_to_plot = results_to_plot[columns]
		for pane_ind, logy in enumerate([False, True]):
			ax = results_to_plot.plot(ax=ax_arr[pane_ind], logy=logy)
			ax.set_ylabel(ylabel)
			if logy:
				ax.get_legend().remove()
		return ax_arr

	def summary(self, hyperparams=False, G_attrs=False, plot=False, **plot_kwargs):
		"""Print several summary stats, and potentially plot the trends."""
		summary_dict = {attr_name: getattr(self, attr_name) for attr_name in SUMMARY_ATTRS}
		summary_series_list = [pd.Series(summary_dict)]
		if hyperparams:
			summary_series_list.append(pd.Series(self.hyperparams))
		if G_attrs:
			summary_series_list.append(pd.Series(self.G_attrs))
		print(pd.concat(summary_series_list))
		if plot:
			self.plot_trends(hyperparams=hyperparams, G_attrs=G_attrs, **plot_kwargs)

