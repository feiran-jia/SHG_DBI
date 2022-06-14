
import numpy as np
from src.gradient import Gradients
from tqdm import tqdm
import copy
import time

class GradDescent:
	'''
	Methods:
		update:			perform a single interaction
		performIter: 	perform # maxIter iteractions


		get_grad:		return a list of current gradient information, [array[], array[], ....]
		get_obj:		return a list of objective values, [X,X,...]
		get_iter:		return current iter
		get_actions:	return current actions

		get_hession:	return a list of current hession information

	'''


	def __init__(self, compute_gradient, cost_function, action_profile, **params):
		'''Set Functionss
		'''
		# cost function, a list of cost functions
		self.cost_function = cost_function
		# gradient function
		self.compute_gradient = compute_gradient
		# Record the total_gradient: the way to compute the exact total gradient, serve as evaluation
		self.total_gradient = params["total_gradient"] if "total_gradient" in params else None


		'''Set Actions, Initialization
		'''
		# number of agents
		self.N = len(action_profile) 
		# init the action_profile, a list of N arrays (1 \times dim)
		self.action_profile = action_profile
		# init the grad of current profile, a list of N arrays (1 \times dim)
		self.action_grads = None
		# current iter, initialized to 0
		self.current_iter = 0
		# whether meet the stopping crit
		self.stop = False
		# keep a best strategy
		self.best_action = copy.deepcopy(action_profile)
		self.min_norm = float('inf')

		'''Set Paramss
		'''
		# Stopping criteria 
		self.stop_grad = params["stop_grad"] if "stop_grad" in params else 1e-8
		# learning rate for all the agents, a list of N
		self.lr = params["lr"] if 'lr' in params else [0.01 for i in range(self.N)]
		# upper bound of the iteraction
		self.max_iter = params["max_iter"] if 'max_iter' in params else 4000
		# how to compute the norm
		self.which_norm = params["which_norm"] if "which_norm" in params else "L2"
		# whether the keep the history
		self.if_trace = params["if_trace"] if "if_trace" in params else True
		# whether to check the 2nd order condition
		self.if_check_2nd = True if "second_order" in params else False
		# calculate the 2nd order condition
		self.second_order = params["second_order"] if "second_order" in params else None
		# random_start_range [0] is lower bound, [1] is higher bound
		self.random_start_range = params["random_start_range"] if "random_start_range" in params else [0,1]


		if self.if_trace:
			self.norm_trace = []
			self.total_gradient_norm_trace = [] if "total_gradient" in params else None
			self.time_trace = [] # time for each iteration
			self.profile_trace = [] # action profile for each iteration

	def perform_iter(self):

		# Initialize iteration variables
		delta = float("inf")

		# Run algorithm
		for i in tqdm(range(self.max_iter)):
			# If the stop crit is met, stop the iter
			if self.stop:
				break


			start = time.time()
			# Update the iter
			self.current_iter = i + 1

			# Perform a descent step
			self.update()
			
			end = time.time()

			time_for_iter = end - start

			if self.if_trace:
				self.time_trace.append(time_for_iter)
				self.profile_trace.append(copy.deepcopy(self.action_profile))
			# Evaluating the stopping criteria
			self.stop_crit()


	def update(self):
		'''
		One iter of Gradient Descent
		'''
		# compute a list of grads regarding the current action_profile
		# update the action_grads
		self.action_grads = self.compute_gradient(self.action_profile)
		# update
		self.action_profile = [self.action_profile[i] - self.lr[i] * self.action_grads[i] for i in range(self.N)]



	def stop_crit(self):
		'''
		Check the stopping criteria
		'''
		if self.which_norm == "Avg":
		# compute the norm of each agent, F-norm, 2-norm
			grads_norms = [np.linalg.norm(self.action_grads[i]) for i in range(self.N)]
			# compute the average norm
			grads_norm = np.mean(grads_norms)
		elif self.which_norm == "L2":
			grad_list = [self.action_grads[i].flatten() for i in range(self.N)]
			# Flatten the norm
			grad_list = np.concatenate(grad_list)
			# compute the norm
			grads_norm = np.linalg.norm(grad_list)
		else:
			assert False, "Error in stop_crit"

		# record the best strategy
		if grads_norm < self.min_norm:
			self.min_norm = grads_norm
			self.best_action = copy.deepcopy(self.action_profile)

		# keep a history of the grad norms
		if self.if_trace:
			self.norm_trace.append(grads_norm)

			# keep the history of the true total gradient norm
			if self.total_gradient_norm_trace is not None:
				_total_gradient = self.total_gradient(self.action_profile)
				_total_grad_list = [_total_gradient[i].flatten() for i in range(self.N)]
				_total_grad_list = np.concatenate(_total_grad_list)
				_total_grad_norm = np.linalg.norm(_total_grad_list)
				self.total_gradient_norm_trace.append(_total_grad_norm)

		# stop when satisfies 1st 2nd conditions
		# 1st condition
		if grads_norm <= self.stop_grad:
			self.stop = True
			# 2nd condition
			if self.if_check_2nd:
				self.stop = True if self.check_2nd_condition() else False
			
			if self.stop == False:
				print("second_order is not satisfied")
				lb = self.random_start_range[0]
				ub = self.random_start_range[1]
				self.action_profile = [np.random.uniform(lb, ub, self.action_profile[i].shape[1]) for i in range(self.N)]
				print("random start to ", self.action_profile)



	def check_2nd_condition(self):
		'''This function can just check 2nd condition for 1-dim action
		'''
		_second_order_factor = self.second_order(self.action_profile)
		# if there exists negative values, set it to True
		_negative_value_in_2nd = np.any(np.stack(_second_order_factor)<0)
		return not _negative_value_in_2nd

	def get_grad(self):
		return self.action_grads


	def get_obj(self):
		'''
		return a list of arraies, N
		'''
		return self.cost_function(self.action_profile) 

	def get_actions(self):
		return self.action_profile


	def get_iter(self):
		return self.current_iter

	def get_norm_trace(self):
		'''
		return a list of norm history
		'''
		return self.norm_trace

	def get_total_gradient_norm_trace(self):
		'''
		return a list of norm history
		'''
		return self.total_gradient_norm_trace

	def get_best_action(self):
		''' 
		return the best action we found up to now
		'''
		return self.best_action

	def get_profile_trace(self):
		return self.profile_trace

	def get_time_trace(self):
		return self.time_trace


	def learning_schedule(self):
		# To-Do: change the learning rates
		pass

class ProjGradDescent(GradDescent):
	'''
	Project Grad Descent
	'''
	def __init__(self, compute_gradient, cost_function, action_profile, projector = None, interval_space = [0, 1], **params):
		super(__class__, self).__init__(compute_gradient, cost_function, action_profile, **params)
		# Projected Function
		self.projector = projector if projector else self.interval_projector
		self.interval_space = interval_space

		# whether stop when converges: might converge to the bundaries
		self.check_convergence = params["check_convergence"] if "check_convergence" in params else False

	def update(self):
		'''
		One iter of Gradient Descent
		'''
		# compute a list of grads regarding the current action_profile
		# update the action_grads
		self.action_grads = self.compute_gradient(self.action_profile)
		# update
		self.action_profile = [self.projector(self.action_profile[i] - self.lr[i] * self.action_grads[i]) for i in range(self.N)]

	def interval_projector(self, profile):
		# If there is no projector, default to projected to [0, 1]

		return np.clip(profile, self.interval_space[0], self.interval_space[1])


	def stop_crit(self):
		'''
		Check the stopping criteria
		TO-DO: need to check it
		'''
		# To-Do: we can have more stopping criteria
		if self.which_norm == "Avg":
		# compute the norm of each agent, F-norm, 2-norm
			grads_norms = [np.linalg.norm(self.action_grads[i]) for i in range(self.N)]
			# compute the average norm
			grads_norm = np.mean(grads_norms)
		elif self.which_norm == "L2":
			grad_list = [self.action_grads[i].flatten() for i in range(self.N)]
			# Flatten the norm
			grad_list = np.concatenate(grad_list)
			# compute the norm
			grads_norm = np.linalg.norm(grad_list)
		else:
			assert False, "Error in stop_crit"

		# record the best strategy
		# if grads_norm < self.min_norm:
		# 	self.min_norm = grads_norm
		# 	self.best_action = copy.deepcopy(self.action_profile)

		if self.check_convergence:
			if self.best_action == self.action_profile:
				self.stop = True


		self.best_action = copy.deepcopy(self.action_profile)

		if self.if_trace:
			self.norm_trace.append(grads_norm)

			# keep the history of the true total gradient norm
			if self.total_gradient_norm_trace is not None:
				_total_gradient = self.total_gradient(self.action_profile)
				_total_grad_list = [_total_gradient[i].flatten() for i in range(self.N)]
				_total_grad_list = np.concatenate(_total_grad_list)
				_total_grad_norm = np.linalg.norm(_total_grad_list)
				self.total_gradient_norm_trace.append(_total_grad_norm)

		if grads_norm < self.stop_grad:
			self.stop = True

		







