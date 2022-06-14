import numpy as np
# this is the config file for SecGame experiments
# structure: 1-3-6

# PGD execution
max_iter = 10001
learning_rate = 0.1

# initial point
alpha_g = np.array([[0.5]])
alpha_s = [np.array([[0.5]]), np.array([[0.5]]), np.array([[0.5]])]
alpha_m = [np.array([[0.5]]) for i in range(6)]

# BRD execution
max_rounds = [2,2,2]
discretization_factors = [0.01, 0.01, 0.01]
upper_bound = 1.0

# BRD evaluation: use BRD to evaluate 
max_rounds_eva = [3,3,3]
discretization_factors_eva = [0.05, 0.05, 0.05]
upper_bound_eva = 2.0
num_sample = 1

# setting for the kappa
kappa_s = 0.1
kappa_m = 0.1

# setting for sec game
# Q: influence factor
num_last_level = 6
Q = (1.0/num_last_level)*np.ones((num_last_level,num_last_level))
#Q[0] = [1, 0,0,0,0,0]
#Q[5] = [0,0,0,0,0,1]
lambda_a = 5
cost_factor = 0.2

# setting for the code
verbose = True # print some intermediate results


# random seed
seed = 0



