
import os
import numpy as np
import torch
from hessian import hessian

from src.gradient import Gradients
from src.grad_algs import *
from src.HBRD import HBrd
from src.payoff_func import Payoff

import pickle
import time 
from munch import Munch
from utils.config.config import Config
from utils.parse import get_parser

''' Data set
'''
def build_environment():
	# network

	n_l = 6 + 3 + 1 # The number of the agents

	# a list of integers specifying agents' action dimensions; also provide agent index

	dimensions = [1 for i in range(n_l)] 

	# level_sizes: a list of intergers specifying the size of each level; we assume agent 0 are the root so level_sizes[0] = 1; the agents are indexed via level-order traverse

	level_sizes = [1, 3, 6]

	# parents: a list of integers pointing out the index of one's direct parent; assune parents[0] = -1

	parents = [0] + [0, 0, 0] + [1,1,2,2,3,3]

	bottom_idx = list(range(4,n_l))

	region_dict = {0:[0,1], 1:{2,3}, 2:{4,5}}

	return n_l, dimensions, level_sizes, parents, bottom_idx, region_dict

''' Payoff Functions of HSecG
'''

def payoff_func_m_generator(m, Q, kappa_m = 0.5, lambda_a = 1, cost_factor = 1):
    # L = 3
    Q = torch.from_numpy(Q)

    def payoff_func_m(alpha, alpha_p, alpha_M):

        # a_m: defender's strategy
        a_m = alpha_M.flatten()
        mask = torch.zeros(*(a_m.size())).bool()
        mask[m] = True
        a_m = torch.where(mask, alpha[0, 0], a_m)
        
        # a: attacker's strategy softmax(lambda (1-a_m))
        a = torch.nn.Softmax(dim = 0)(lambda_a * (1-a_m)) 
        
        # utility of the defender
        u_m = - cost_factor * a_m[m] + a[m] * a_m[m] / (1 + a_m[m])
        for i in range(len(a_m)):
            if i != m:
                u_m += a[i] * (1 - Q[m,i] * 1 / ((1 + a_m[m]) * (1 + a_m[i]))) 
        c_m =  - kappa_m * u_m + (1 - kappa_m) *(alpha-alpha_p)*(alpha-alpha_p)
        
        return c_m

    return payoff_func_m


def payoff_func_s_generator(s, Q, region_dict, kappa_s = 0.5, lambda_a = 1, cost_factor = 1):
	Q = torch.from_numpy(Q)

	def payoff_func_s(alpha, alpha_p, alpha_M):


		# init the strategies 
		a_m = alpha_M.flatten()
		a = torch.nn.Softmax(dim = 0)(lambda_a * (1-a_m)) 

		# overall utility of its members
		GSW = torch.tensor([[0.0]])
		for m in region_dict[s]:

			i_m = - cost_factor * a_m[m] + a[m] * a_m[m] / (1 + a_m[m])
			for i in range(len(a_m)):
			    if i != m:
			        i_m += a[i] * (1 - Q[m,i]  / ((1 + a_m[m]) * (1 + a_m[i]))) 
			GSW += i_m

		c_s = - kappa_s * GSW + (1 - kappa_s) *(alpha-alpha_p)*(alpha-alpha_p)

		return c_s
	return payoff_func_s

def payoff_func_g_generator(Q, lambda_a = 1, cost_factor = 1):
	Q = torch.from_numpy(Q)


	def payoff_func_g(alpha, alpha_p, alpha_M):

	    # init the strategies 
	    
	    a_m = alpha_M.flatten()
	    a = torch.nn.Softmax(dim = 0)(lambda_a * (1-a_m)) 
	    SW = torch.tensor([[0.0]])
	    for m in range(len(a_m)):
	        i_m = - cost_factor * a_m[m] + a[m] * a_m[m] / (1 + a_m[m])
	        for i in range(len(a_m)):
	            if i != m:
	                i_m += a[i] * (1 - Q[m,i]  / ((1 + a_m[m]) * (1 + a_m[i]))) 
	        SW += i_m
	    
	    c_g = - SW + 0.0 * alpha
	    return c_g

	return payoff_func_g

if __name__ == '__main__':
	# args
	parser = get_parser()
	args = parser.parse_args()
	mode = args.mode
	cfg_name = args.cfg_name

	# cfgs
	cfg1 = Config.fromfile('exp/exp_configs/secg/{}.py'.format(cfg_name))
	cfg = Munch(cfg1)

	# random seed
	np.random.seed(cfg.seed)


	# init data
	n_l, dimensions, level_sizes, parents, bottom_idx, region_dict = build_environment()

	# starting point 
	alpha_g = cfg.alpha_g
	alpha_s = cfg.alpha_s
	alpha_m = cfg.alpha_m

	_action_profile = [alpha_g]+ alpha_s + alpha_m

	# Setting Payoff functions

	payoff_funcs = []
	self_gradient_funcs = []
	bottom_gradient_funcs = []
	child_to_parent_hessian_funcs = []


	# funcs for G level

	payoff_g = Payoff(payoff_func_g_generator(cfg.Q, lambda_a = cfg.lambda_a,\
			 cost_factor = cfg.cost_factor),  bottom_idx)
	payoff_funcs.append(payoff_g.get_payoff_func)
	self_gradient_funcs.append(payoff_g.get_gradient_func)
	bottom_gradient_funcs.append(payoff_g.get_bottom_grad_func)
	child_to_parent_hessian_funcs.append(payoff_g.get_child_to_parent_hessian_func)


	# funcs for S level
	for s in [0,1,2]:
		payoff_s = Payoff(payoff_func_s_generator(s, cfg.Q, region_dict,\
			 kappa_s = cfg.kappa_s, lambda_a = cfg.lambda_a, cost_factor = cfg.cost_factor), bottom_idx)
		payoff_funcs.append(payoff_s.get_payoff_func)
		self_gradient_funcs.append(payoff_s.get_gradient_func)
		bottom_gradient_funcs.append(payoff_s.get_bottom_grad_func)
		child_to_parent_hessian_funcs.append(payoff_s.get_child_to_parent_hessian_func)


	# funcs for M level
	for m in range(6):

		payoff_m = Payoff(payoff_func_m_generator(m, cfg.Q, kappa_m = cfg.kappa_m,\
			 lambda_a = cfg.lambda_a, cost_factor = cfg.cost_factor), bottom_idx)
		payoff_funcs.append(payoff_m.get_payoff_func)
		self_gradient_funcs.append(payoff_m.get_gradient_func)
		bottom_gradient_funcs.append(payoff_m.get_bottom_grad_func)
		child_to_parent_hessian_funcs.append(payoff_m.get_child_to_parent_hessian_func)


	# start optimizing, algorithm: PGG
	ig = Gradients(dimensions, level_sizes, parents, payoff_funcs,\
	              self_gradient_funcs=self_gradient_funcs,\
	              bottom_gradient_funcs=bottom_gradient_funcs,\
	              child_to_parent_hessian_funcs=child_to_parent_hessian_funcs)


	if mode == "test":
		# test the code
		brd = HBrd(parents, level_sizes, payoff_funcs, _action_profile,if_detect_cycle = True, \
			max_rounds = cfg.max_rounds, discretization_factors = cfg.discretization_factors)
		res = brd.eva_spe_epsilon(_action_profile)
		print(res)

	elif mode == "PGD":
		pgd = ProjGradDescent(ig.implicit_gradients, ig.get_payoffs, _action_profile,  max_iter = cfg.max_iter,\
						 stop_grad = 0.0, lr = [cfg.learning_rate for i in range(n_l)], interval_space = [0, np.infty])
		pgd.perform_iter()
		profile_history = pgd.get_profile_trace()
		time_trace = pgd.get_time_trace()
		norm_trace = pgd.norm_trace

		if cfg.verbose:
			print("last norm = ", norm_trace[-1])
			print("last profile = ", profile_history[-1])

		# store the PGG results

		dir_path = './res/sec_results/'
		if(not os.path.exists(dir_path)):
			os.makedirs(dir_path)
		res_name = 'pgd_trace_{}.pkl'.format(cfg_name)

		with open(dir_path+res_name, "wb") as f:
		    pickle.dump((time_trace, norm_trace, profile_history), f)

	elif mode == "Eva_PGD":
	   
		''' Evaluate the PGG with brd
		'''

		# load profile_history
		file =  "./res/sec_results/pgd_trace_{}.pkl".format(cfg_name)
		time_trace, _, profile_history = pickle.load(open(file, 'rb'))


		brd = HBrd(parents, level_sizes, payoff_funcs, _action_profile, if_detect_cycle = True, max_rounds = cfg.max_rounds_eva,\
		 discretization_factors = cfg.discretization_factors_eva)
		
		epsilon_list = []
		run_time_list = []
		sample = np.linspace(0, cfg.max_iter, cfg.num_sample)
		
		for idx in sample:
			run_time = sum(time_trace[0:int(idx) + 1])
			re_epsilon = brd.eva_hie_epsilon(profile_history[int(idx)])
			epsilon_list.append(re_epsilon)
			run_time_list.append(run_time)
		if cfg.verbose:
			print(epsilon_list)
			print("run_time_list = ", run_time_list)
		
		dir_path = './res/sec_results/'
		if(not os.path.exists(dir_path)):
			os.makedirs(dir_path)
		res_name = 'pgd_epsilon_{}.pkl'.format(cfg_name)

		with open(dir_path + res_name, "wb") as f:
			pickle.dump((sample, run_time_list, epsilon_list),f)

	elif mode == "BRD":
		''' BRD
		'''
		brd = HBrd(parents, level_sizes, payoff_funcs, _action_profile,if_detect_cycle = True, \
			max_rounds = cfg.max_rounds, discretization_factors = cfg.discretization_factors, upper_bound = cfg.upper_bound)
		start = time.time()
		#file_name = "./exp_results/pgg_results/brd_trace_{}.pkl".format(cfg_name)


		best_action_profile, epsilon = brd.perform_brd(_action_profile) 
		end = time.time()

		brd_time = end - start

		if cfg.verbose:
			print("best_action_profile = ", best_action_profile)
			print("epsilon = ", epsilon)
			print("BRD_time = ", brd_time)


		# the file of brd trajectory
		dir_path = './res/sec_results/'
		if(not os.path.exists(dir_path)):
			os.makedirs(dir_path)
		res_name = 'brd_trace_{}.pkl'.format(cfg_name)

		with open(dir_path + res_name, 'wb') as f:
			pickle.dump((brd_time, best_action_profile), f)

	elif mode == "Eva_BRD":

		'''Evaluating BRD
		'''
		file =  "./res/sec_results/brd_trace_{}.pkl".format(cfg_name)
		brd_time, best_action_profile = pickle.load(open(file, 'rb'))
		
		brd = HBrd(parents, level_sizes, payoff_funcs, _action_profile,if_detect_cycle = True,\
			 max_rounds = cfg.max_rounds_eva, discretization_factors = cfg.discretization_factors_eva, upper_bound = cfg.upper_bound_eva)
		re_epsilon = brd.eva_hie_epsilon(best_action_profile)

		dir_path = './res/sec_results/'
		if(not os.path.exists(dir_path)):
			os.makedirs(dir_path)
		res_name = 'brd_epsilon_{}.pkl'.format(cfg_name)

		with open(dir_path + res_name, "wb") as f:
			pickle.dump((brd_time, re_epsilon),f)

		if cfg.verbose:
			print("brd_time =", brd_time)
			print("epsilon is ", re_epsilon)
	else:
		print("We have 5 mode: test, PGD. BRD, Eva_PGD, and Eva_BRD.")