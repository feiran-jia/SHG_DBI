import numpy as np
# this is the config file for PGG experiments
# network: karate_club_graph

# PGD execution
max_iter = 10001
learning_rate = 0.1

# initial point
alpha_g = np.array([[0.5]])
alpha_s = [np.array([[0.5]]), np.array([[0.5]])]
alpha_m = [np.array([[0.5]]) for i in range(34)]

# BRD execution
max_rounds = [1,1,1]
discretization_factors = [0.1, 0.1, 0.1]

# BRD evaluation: use BRD to evaluate 
max_rounds_eva = [3,3,3]
discretization_factors_eva = [0.05, 0.05, 0.05]
num_sample = 1

# setting for the kappa
kappa_s = 0.5
kappa_m = 0.5

# setting for Karate_club_network
a = 0
b = 1 
c = 6

# setting for the code
verbose = True # print some intermediate results


# random seed
seed = 0



