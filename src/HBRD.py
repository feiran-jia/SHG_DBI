import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import copy
from multiprocessing import Pool
import os
import pickle
import datetime
import time
from itertools import product




class HBrd:
    """docstring for HBRD
        each agent just 1-dim action profile
    """
    def __init__(self, parents, level_sizes, payoff_funcs, action_profile, **params):
        # number of agents
        self.N = len(payoff_funcs)
        #parents: a list of integers pointing out the index of one's direct parent; assume parents[0] = 0
        self.parents = parents
        self.level_sizes = level_sizes
        self.num_of_levels = len(level_sizes)
        self.L = self.num_of_levels - 1 # the idx of last level
        self.level_to_idxs = self._init_num_of_levels()
        self.idxs_to_level = self._init_idxs_to_level()
        self.leaves, self.children = self._init_leaves_and_children()
        self.payoff_funcs = payoff_funcs
        self.brd_counter = 0
        self.brd_profile_trajectory = []
        self.if_record_trajectory = True # if eva epsilon, set to False


        '''Unpack and initialize all settings'''
        self.discretization_factors = params['discretization_factors'] if 'discretization_factors' in params else [0.1 for i in range(self.N)]
        self.upper_bound = params['upper_bound'] if 'upper_bound' in params else 1.0
        self.EPSILON = params['EPSILON'] if 'EPSILON' in params else 0.0000001    ## General stopping criterion for BRD
        self.max_rounds = params['max_rounds'] if 'max_rounds' in params else [100 for i in range(self.N)]
        self.if_detect_cycle = params['if_detect_cycle'] if 'if_detect_cycle' in params else True
        self.brd_iter_num = params['brd_iter_num'] if 'brd_iter_num' in params else 1


    def _init_num_of_levels(self): 
        level_to_idxs = []
        cur_idx = 0
        for l in range(self.num_of_levels):
            level_to_idxs.append(list(range(cur_idx, cur_idx+self.level_sizes[l])))
            cur_idx += self.level_sizes[l]
        return level_to_idxs

    def _init_idxs_to_level(self): 
        idxs_to_level = list(range(self.N))
        for l in range(self.num_of_levels):
            idxs_in_level_l = self.level_to_idxs[l]
            for idx in idxs_in_level_l:
                idxs_to_level[idx] = l
        return idxs_to_level

    def _init_leaves_and_children(self): 
        # the leaves agent influenced
        leaves = [[] for n in range(self.N)]
        # one's direct children
        children = [[] for n in range(self.N)]

        for n in range(self.N-1, 0, -1):
            if(n in self.level_to_idxs[self.num_of_levels-1]):
                leaves[n].append(n)

            leaves[self.parents[n]] += leaves[n]
            children[self.parents[n]].append(n)
        return leaves, children

    def grid_search(self, action_profile, agent_idx, l):
        action_profile = copy.deepcopy(action_profile)
        # init the best profile
        best_payoff = float('inf')
        best_action = action_profile[agent_idx]
        best_action_profile = copy.deepcopy(action_profile)
        best_epsilon_of_descendents = float('inf')

        for current_action in tqdm(np.linspace(0,self.upper_bound,int(np.ceil(1+self.upper_bound/self.discretization_factors[l])))):

            current_action_profile = action_profile.copy()
            current_action_profile[agent_idx] = np.array(current_action).reshape(1,1)
            self_action = current_action_profile[agent_idx]

            payoff, epsilon_of_descendents, current_action_profile = self.payoff_re_eq(l, agent_idx, current_action_profile)

            if payoff < best_payoff:
                best_action = current_action
                best_payoff = payoff
                best_action_profile = copy.deepcopy(current_action_profile)
                best_epsilon_of_descendents = epsilon_of_descendents

        return best_action, best_payoff, best_epsilon_of_descendents, best_action_profile

    def payoff_re_eq(self, l, agent_idx, action_profile):
        # init epsilon_of_descendents  = 0, if L = l, set the epsilon_of_descendent = 0
        epsilon_of_descendents = 0
        # if l is not the last level, we re-eq the descendents
        action_profile = copy.deepcopy(action_profile)
        if l < self.L:
            brd_set = self.children[agent_idx]
            action_profile, epsilon_of_descendents = self.HG_PSPNE(l + 1, brd_set, action_profile)
        payoff = self.payoff_func(action_profile, agent_idx)
        return payoff, epsilon_of_descendents, action_profile

    def random_start(self, brd_set, action_profile, l):
        discrete_actions = np.linspace(0,self.upper_bound,int(np.ceil(1+self.upper_bound/self.discretization_factors[l])))
        for agent in brd_set:
            action_profile[agent] = np.random.choice(discrete_actions,[1,1])
        return action_profile

    def HG_PSPNE(self, l, brd_set, action_profile):
        t = 0
        action_profile = self.random_start(brd_set, action_profile, l)
        # Init best epsilon
        epsilon = float('inf')
        # Init best profile, select the best profile
        best_action_profile = copy.deepcopy(action_profile)
        # print("action_profile =", action_profile)


        _Profile_Trace = [] if self.if_detect_cycle else None

        while (epsilon > self.EPSILON and t < self.max_rounds[l]):
            # cycle detection
            if self.if_detect_cycle:
                while action_profile in _Profile_Trace:
                    # random start
                    action_profile = copy.deepcopy(self.random_start(brd_set, action_profile,l))
                _Profile_Trace.append(copy.deepcopy(action_profile))


            max_epsilon = 0
            brd_actions = {}
            brd_epsilons = {}
            brd_profiles = {}

            old_profile = copy.deepcopy(action_profile)
            for agent_idx in brd_set:
                current_payoff, current_epsilon_of_descendents, current_action_profile = self.payoff_re_eq(l, agent_idx, action_profile)#self.payoff_func(action_profile, agent_idx)
                # the best response
                best_action, best_payoff, epsilon_of_descendents, best_action_profile = self.grid_search(action_profile, agent_idx, l)

                # epsilon of agent_idx itself is calculated by max{epsilon_of_descendents, epsilon_itself}
                epsilon_idx =  current_payoff - best_payoff
                epsilon_idx = max(epsilon_idx, current_epsilon_of_descendents)

                # record the brd actions
                brd_actions[agent_idx] = best_action
                brd_epsilons[agent_idx] = epsilon_idx
                brd_profiles[agent_idx] = copy.deepcopy(best_action_profile)


            # Preform BRD for all the brd_set agents
            new_profile = copy.deepcopy(action_profile)
            for agent_idx in brd_set:
                brd_action_profile = brd_profiles[agent_idx]
                # propogate profiles from the descendents

                for i in range(self.N):
                    if brd_action_profile[i] != action_profile[i]:
                        new_profile[i] = brd_action_profile[i]

                new_profile[agent_idx] = np.array(brd_actions[agent_idx]).reshape(1,1)

                # propogate the epsilon of agent_idx's descendents
                if brd_epsilons[agent_idx] > max_epsilon:
                    max_epsilon = brd_epsilons[agent_idx]

            if self.if_record_trajectory and (self.brd_counter % self.brd_iter_num == 0):
                old_time = self.time
                self.time = time.time()
                self.brd_profile_trajectory.append((copy.deepcopy(action_profile), self.time-old_time))
                self.brd_counter += 1

            if max_epsilon < epsilon:
                epsilon = max_epsilon
                best_action_profile = copy.deepcopy(action_profile)#action_profile) #debug here
                if l == 0:
                    print("best_action_profile = ", best_action_profile)

            # Update the time step
            t += 1
            action_profile = copy.deepcopy(new_profile)
            print("t = ", t)

        return best_action_profile, epsilon

    def payoff_func(self, action_profile, agent_idx):
        # current payoff of agent agent_idx
        self_action = action_profile[agent_idx]
        parent_action = action_profile[self.parents[agent_idx]]
        bottom_actions = [action_profile[i] for i in self.level_to_idxs[self.L]]
        current_payoff = self.payoff_funcs[agent_idx](self_action, parent_action, bottom_actions).item()
        return current_payoff

    def perform_brd(self, action_profile, name=""):
        # start from the root node
        self.time = time.time()
        res =  self.HG_PSPNE(0, [0], action_profile)
        # with open(name, "wb") as f:
        #     pickle.dump(self.brd_profile_trajectory, f)
        self.brd_profile_trajectory = []
        self.brd_counter = 0
        return res

    def compute_regret(self, action_profile, agent_idx):
        ''' Compute the regret of agent agent_idx
            Consider the Best Respose without Re-eq
            Using Grid Search
        '''
        # payoff of original profile
        payoff = self.payoff_func(action_profile, agent_idx)

        # Best Response
        best_payoff = float('inf')
        l = self.idxs_to_level[agent_idx]
        for current_action in tqdm(np.linspace(0,self.upper_bound,int(np.ceil(1+self.upper_bound/self.discretization_factors[l])))):
            # unilateral deviaiton
            new_action_profile = copy.deepcopy(action_profile)
            new_action_profile[agent_idx] = np.array(current_action).reshape(1,1)

            new_payoff = self.payoff_func(new_action_profile, agent_idx)

            if new_payoff < best_payoff:
                best_payoff = new_payoff

        # Regret
        regret = payoff - best_payoff

        return regret

    def eva_nash_epsilon(self, action_profile):
        '''Given an action, return the epsilon for NE of the profile
        '''
        epsilon = 0

        for agent_idx in range(self.N):
            regret = self.compute_regret(action_profile, agent_idx)
            if regret > epsilon:
                epsilon = regret
        return epsilon

    def enumerate_profile(self, l, action_profile):
        # enumate the profile in

        grids = np.linspace(0,self.upper_bound,int(np.ceil(1+self.upper_bound/self.discretization_factors[l])))
        combinations = list(product(grids, repeat = self.level_sizes[l]))
        COMBS = []
        for comb in combinations:
            new_profile = copy.deepcopy(action_profile) #[x.clone() for x in action_profile]
            idxs = self.level_to_idxs[l]

            for i in range(self.level_sizes[l]):
                new_profile[idxs[i]] = np.array(comb[i]).reshape(1,1)

            COMBS.append(new_profile)
        return COMBS


    def eva_spe_epsilon(self, action_profile):
        '''Given an action, return the epsilon of the spe
        The straightforward way: enumerate every possible action combination among 1~l-1 when we are considering layer l. 
        Due to the utility structure in terms of computing maximum deviation 
            - we only need to enumerate the action combination of layer l-1. 
            - so for each such enumeration we find the maximum unilateral deviation in payoffs of players at l~L. 
            - assuming lower-level players are fixed.

        To compute the epsilon effeciently,
            - we compute the epsilon level by level
            1. Enumerate l-1
            2. Compute epsilon for l
        '''
        epsilon = 0
        # l = 0
        for agent_idx in self.level_to_idxs[0]:
            regret = self.compute_regret(action_profile, agent_idx)
            if regret > epsilon:
                epsilon = regret

        # l = 1 - L-1
        for l in range(1, self.L+1):
            print("l = ", l)
            COMBS = self.enumerate_profile(l-1, action_profile)
            for profile in COMBS:
                for agent_idx in self.level_to_idxs[l]:
                    regret = self.compute_regret(profile, agent_idx)
                    if regret > epsilon:
                        epsilon = regret
        return epsilon
            
        




    def eva_hie_epsilon(self, action_profile):
        '''Given an action profile, return the hie_epsilon of the profile
        '''
        self.if_record_trajectory = False
        epsilon = 0
        for agent_idx in range(self.N):
            l = self.idxs_to_level[agent_idx]
            # payoff of original profile
            current_payoff, current_epsilon_of_descendents, current_action_profile = self.payoff_re_eq(l, agent_idx, action_profile)#self.payoff_func(action_profile, agent_idx)
            # the best response
            
            best_action, best_payoff, epsilon_of_descendents, best_action_profile = self.grid_search(action_profile, agent_idx, l)
            epsilon_agent = current_payoff - best_payoff
            #epsilon_agent = max{epsilon_agent, current_epsilon_of_descendents} 

            if epsilon_agent > epsilon:
                epsilon = epsilon_agent
                # print("agent ", agent_idx,"'s epsilon is ",epsilon)
                # print("current_payoff = ", current_payoff)
                # print("best_payoff = ", best_payoff)
                # print("action_profile = ", action_profile)
            if l == 0:
                print("current_payoff = ", current_payoff)
                print("best_payoff = ", best_payoff)

        return epsilon


    def eva_hie_epsilon_trick(self, action_profile, best_payoff_root):
        '''Given an action profile, return the hie_epsilon of the profile
            This trick is built for the 2-level problem 
        '''
        self.if_record_trajectory = False
        epsilon = 0
        for agent_idx in range(self.N):
            l = self.idxs_to_level[agent_idx]
            current_payoff, current_epsilon_of_descendents, current_action_profile = self.payoff_re_eq(l, agent_idx, action_profile)#self.payoff_func(action_profile, agent_idx)
            
            if agent_idx == 0:
                best_payoff = best_payoff_root
            else:
                best_action, best_payoff, epsilon_of_descendents, best_action_profile = self.grid_search(action_profile, agent_idx, l)
            epsilon_agent = current_payoff - best_payoff

            if epsilon_agent > epsilon:
                epsilon = epsilon_agent
                
        return epsilon

class SHGBrd(HBrd):
    ''' A heuristic algorithm for structured Hieratical Game
    '''
    def __init__(self, parents, level_sizes, payoff_funcs, action_profile, **params):
        super(__class__, self).__init__(parents, level_sizes, payoff_funcs, action_profile, **params)
        self.max_rounds = [2 for i in range(self.N)]
        self.eva_max_rounds = params['max_rounds'] if 'max_rounds' in params else [1000 for i in range(self.N)]

        self.T = params['T'] if 'T' in params else 100000

        # Record the epsilon
        # by default do not calculate the epsilon
        self.check_t_list = params['check_t_list'] if 'check_t_list' in params else [] #list(range(0,T,T//10))
        self.epsilon_trace = [] # should have the same length with self.check_t_list


    def eva_hie_epsilon(self, action_profile):
        '''Given an action profile, return the hie_epsilon of the profile
        '''
        self.max_rounds = copy.deepcopy(self.eva_max_rounds)
        epsilon = 0
        for agent_idx in range(self.N):
            # payoff of original profile
            current_payoff = self.payoff_func(action_profile, agent_idx)
            # the best response
            l = self.idxs_to_level[agent_idx]
            best_action, best_payoff, epsilon_of_descendents, best_action_profile = self.grid_search(action_profile, agent_idx, l)
            epsilon_agent = current_payoff - best_payoff

            # epsilon_agent = max{epsilon_agent, current epsilon_of_descendents}
            if epsilon_agent > epsilon:
                epsilon = epsilon_agent
                print("agent ", agent_idx,"'s epsilon is ",epsilon)
        self.max_rounds = [1 for i in range(self.N)]
        return epsilon


    def perform_brd(self, action_profile):
        #grid_search(self, action_profile, agent_idx, l)
        for t in tqdm(range(self.T)):

            # check epsilon
            if t in self.check_t_list:
                epsilon = self.eva_hie_epsilon(action_profile)
                self.epsilon_trace.append(epsilon)

            new_profile = copy.deepcopy(action_profile)
            for agent_idx in range(self.N):
                l = self.idxs_to_level[agent_idx]
                best_action, best_payoff, epsilon_of_descendents, best_action_profile = self.grid_search(action_profile, agent_idx, l)
                # print("best_action = ", best_action, "  best_action_profile = ", best_action_profile)
                # print("epsilon_of_descendents = ", epsilon_of_descendents)
                new_profile[agent_idx] = np.array(best_action).reshape(1,1)


            #Converge
            print("action_profile = ", action_profile)
            print("new_profile = ", new_profile)

            gap = self.difference(action_profile, new_profile)
            # print("gap = ", gap)
            if gap < self.EPSILON:
                return action_profile

            action_profile = new_profile
        return action_profile


    def difference(self, a_old, a_new):

        '''Returns L1 norm of the difference'''
        abs_list = [abs(a_old[i][0,0]-a_new[i][0,0]) for i in range(len(a_old))]
        # print("sum(abs_list) = ", sum(abs_list))
        return max(abs_list)

    def get_epsilon_trace(self):
        return self.epsilon_trace
