import numpy as np
N = 10000
MIN_DEGREE = 2
MEAN_DEGREE = 20
GAMMA = 0.0
STEPS_PER_DAY = 5
MAX_STEPS = 3000
INITIAL_INFECTED_NUM = 10
PROB_INFECT = 0.027
PROB_INFECT_EXPOSED_FACTOR = 0.5
# The same ratio holds for the final contagious part of the incubation period.
# Set this to 0 to make the Asymptomatic non-infectious.
# Source: Assumed in a few places. This paper might claim 0.5: https://science.sciencemag.org/content/368/6490/489
RELATIVE_INFECTIOUSNESS_ASYMPTOMATIC = 0.5
# How many days before developing symptoms (becoming Infected) an Exposed
# individual is contagious. Questionable evidence, but 1-2 produces a reasonable R0.
DURATION_EXPOSED_INFECTS = 2
# Incubation period duration distribution mean. In days.
# Source: 3rd Imperial College paper: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
INCUBATION_DURATION_MEAN = 5.1
# Incubation period duration distribution standard deviation. In days.
# Source: 3rd Imperial College paper: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
INCUBATION_DURATION_STD = 4.38
# Share of population who would not develop symptoms even when sick. They are
# not detected along with the symptomatic Infected in the testing step, but are
# detected by direct tests (mass testing or when neighbors are tested).
# Source for default: Vo, Italy study and others (See Eric Topol paper): https://www.scripps.edu/science-and-medicine/translational-institute/about/news/sarc-cov-2-infection/index.html
PROB_ASYMPTOMATIC = 0.40
# Probability to change from Infections to Recovered in a single day.
# Equals the inverse of the expected time to Recover.
# 3.5 = 0.3 * 0 + 0.56 * 5 + 0.1 * 5 + 0.04 * 5. Source: Pueyo spreadsheet: https://docs.google.com/spreadsheets/d/1uJHvBubps9Z2Iw_-a_xeEbr3-gci6c475t1_bBVkarc/edit#gid=0
PROB_RECOVER = 1 / 3.5
# Length of imposed quarantine, in days.
DAYS_IN_QUARANTINE = 14
# Probability that an Infected individual is tested and detected as a carrier in a single day.
# This encapsulates also the fact that many people are mildly symptomatic or will
# not be tested (or quarantined).
# Default: no tests at all.
PROB_INFECTED_DETECTED = 0
# Probability that a neighbor of an individual who tested positive is himself traced,
# and then tested / quarantined. The testing of neighbors happens once, once the
# original individual is tested positive, so this is *not* a probability per day.
# Default - no neighbors are traced.
PROB_NEIGHBOR_TRACED = 0
# Share of the general population which is tested on a single day. Exposed and
# Infected individuals who are tested are detected as carriers.
# Default - previous behavior, no exposed are detected.
PROB_EXPOSED_DETECTED = 0
# When an individual tests positive, whether their traced neighbors are quarantined.
QUARANTINE_NEIGHBORS = False
# When an individual tests positive, whether their traced neighbors are tested.
TEST_NEIGHBORS = False
# Delay between the time a test is performed and the time results become known
# to the subject, and contact tracing is complete. Days.
TEST_DELAY_TIME = 0

EPSILON = 1e-10
# USA deaths data (worldometers.info): 11 -> 2220 from March 4th to March 28th.
DOUBLING_DAYS = float((28 - 4) / np.log2(2220 / 11))  # About 3.13
# Names of columns in SimulationResults DataFrame.
MAIN_GROUPS = ['susceptible', 'exposed', 'recovered', 'infected', 'quarantined']
# Column list out of the columns in the SimulationResults DataFrame.
ALL_COLUMNS = MAIN_GROUPS + ['test_rate']
# Plotting color conventions.
GROUP2COLOR = dict(susceptible='blue', exposed='orange', recovered='green',
                   quarantined='purple', infected='red', test_rate='brown')
# Attributes to print in SimulationResults summary.
SUMMARY_ATTRS = ['duration', 'fraction_infected', 'doubling_days', 'fraction_quarantine_time', 'peak_infected_time', 'peak_fraction_infected', 'fraction_tests', 'peak_test_rate']
