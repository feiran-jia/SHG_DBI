import numpy as np
import copy

class Gradients:
    def __init__(self, dimensions, level_sizes, parents, payoff_funcs, self_gradient_funcs=None, bottom_gradient_funcs=None, child_to_parent_hessian_funcs=None, full_jacobian=None):
        '''
            a class implement different gradient dynamics for hierical games

            args:
                dimensions: a list of integers specifying agents' action
                 dimensions; also provide agent index

                level_sizes: a list of intergers specifying the size of each
                 level; we assume agent 0 are the root so level_sizes[0] = 1;
                  the agents are indexed via level-order traverse

                parents: a list of integers pointing out the index of one's
                 direct parent; assume parents[0] = 0

                payoff_funcs: a list of payoff functions; the inputs are
                 (self_action, parent_action, bottom_actions) and return an
                  real number

                self_gradient_funcs: a list of functions return the partial
                 derivatives of one's payoff to its own action; the inputs
                  are (self_action, parent_action, bottom_actions) and return
                   a vector of 1 x dimensions[n]

                bottom_gradient_funcs: a list of functions return the partial
                 derivatives of one's payoff to the bottom-level players'
                  actions; the inputs are (self_action, parent_action,
                   bottom_actions) and return a dictionary containing (key,
                    value) = (k, an 1 x dimensions[k] vector) where k is the
                     idx of bottom-level agents

                child_to_parent_hessian_funcs: a list of functions return the
                 hessian matrix to be propagated between consecutive levels:
                  the -(\nabla^2_{a_{l+1,j}, a_{l+1}, j}u_{l+1, j})^{
                  -1}\nabla^2_{a_{l+1,j}, a_{l,i}}u_{l+1,j}, a d_{l+1, j} x
                   d_{l, i} matrix

                full_jacobian: returns an d * d matrix that is the jacobian
                of the simulatenous learning dynamics: block (i, j) is
                \nabla_{a_j}(D_{a_i}u_i)
        '''
        self.N = len(dimensions)
        assert len(parents) == self.N and len(payoff_funcs) == self.N and len(self_gradient_funcs) == self.N and len(bottom_gradient_funcs) == self.N and len(child_to_parent_hessian_funcs) == self.N
        self.parents = parents
        self.dimensions = dimensions
        self.level_sizes = level_sizes
        self.num_of_levels = len(level_sizes)
        # return the set of agent idxs for a given level
        self.level_to_idxs = []
        cur_idx = 0
        for l in range(self.num_of_levels):
            self.level_to_idxs.append(list(range(cur_idx, cur_idx+self.level_sizes[l])))
            cur_idx += self.level_sizes[l]


        # the leaves agent influenced
        self.leaves = [[] for n in range(self.N)]
        # one's direct children
        self.children = [[] for n in range(self.N)]

        for n in range(self.N-1, 0, -1):
            if(n in self.level_to_idxs[self.num_of_levels-1]):
                self.leaves[n].append(n)

            self.leaves[parents[n]] += self.leaves[n]
            self.children[parents[n]].append(n)



        self.payoff_funcs = payoff_funcs
        self.self_gradient_funcs = self_gradient_funcs
        self.bottom_gradient_funcs = bottom_gradient_funcs
        self.child_to_parent_hessian_funcs = child_to_parent_hessian_funcs
        self.full_jacobian = full_jacobian



    def self_gradients(self, action_profile):
        applied_gradients = [np.zeros((1, self.dimensions[i])) for i in range(self.N)]
        bottom_actions = [action_profile[i] for i in self.level_to_idxs[self.num_of_levels-1]]
        for l in range(self.num_of_levels-1, -1, -1):
            for i in self.level_to_idxs[l]:
                self_action = action_profile[i]
                parent_action = action_profile[self.parents[i]]
                # one's partial gradient
                self_gradient = self.self_gradient_funcs[i](self_action, parent_action, bottom_actions)

                assert self_gradient.shape == (1, self.dimensions[i])

                applied_gradients[i] += self_gradient
        return applied_gradients

    def self_gradients_norm_gradients(self, action_profile, epsilon=0.0001):
        applied_gradients = [np.zeros((1, self.dimensions[i])) for i in range(self.N)]
        bottom_actions = [action_profile[i] for i in self.level_to_idxs[self.num_of_levels-1]]
        for l in range(self.num_of_levels-1, -1, -1):
            for i in self.level_to_idxs[l]:
                for d in range(self.dimensions[i]):
                    action_profile_p1 = copy.deepcopy(action_profile)
                    action_profile_p2 = copy.deepcopy(action_profile)
                    action_profile_p1[i][0][d] += epsilon
                    action_profile_p2[i][0][d] -= epsilon
                    self_gradients_p1 = self.self_gradients(action_profile_p1)
                    self_gradients_p2 = self.self_gradients(action_profile_p2)
                    norm1, norm2 = 0, 0
                    for j in range(self.N):
                        norm1 += np.sum(np.square(self_gradients_p1[j]))
                        norm2 += np.sum(np.square(self_gradients_p2[j]))
                    applied_gradients[i][0][d] += (norm1-norm2)/epsilon
        return applied_gradients

    def consensus_optimization_gradients(self, action_profile, gamma=0.1):
        applied_gradients = [np.zeros((1, self.dimensions[i])) for i in range(self.N)]
        self_gradient = self.self_gradients(action_profile)
        norm_gradient = self.self_gradients_norm_gradients(action_profile)

        for i in range(self.N):

            applied_gradients[i] += self_gradient[i] - gamma * norm_gradient[i]

        return applied_gradients


    def sym_gradients(self, action_profile, align=True):
        '''

            the symplectic gradient dynamics
            args:
                current action profile (a list of numpy arrays) of size N
            returns:
                the corresponding symplectic gradient (a list of numpy arrays)
                 of size N

                symplectic gradient = self gradient + (anti-symmetric(Jac)) *
                self gradient

        '''
        self_gradient = self.self_gradients(action_profile)
        full_jacobian = self.full_jacobian(action_profile)
        assert full_jacobian.shape == (np.sum(self.dimensions), np.sum(self.dimensions))
        anti_sym_full_jacobian = (full_jacobian - full_jacobian.T)/2
        #print('self_gradient!', anti_sym_full_jacobian)
        #assert self_gradient.shape == (1, np.sum(self.dimensions))
        sym_grad = np.concatenate(self_gradient, axis=1) @ anti_sym_full_jacobian

        cur_idx = 0
        sym_gradient = []
        for i in range(self.N):
            sym_gradient.append(sym_grad[:, cur_idx:cur_idx+self.dimensions[i]])
            cur_idx += self.dimensions[i]

        sign = 1
        if(align):
            gradients_norm_gradients = self.self_gradients_norm_gradients(action_profile)
            term1, term2 = 0, 0
            for i in range(self.N):
                term1 += np.sum(self_gradient[i] * gradients_norm_gradients[i])
                term2 += np.sum(sym_gradient[i] * gradients_norm_gradients[i])
            if(term1 * term2/np.sum(self.dimensions) + 1/10 < 0):
                sign = -1

        res_gradient = []
        cur_idx = 0
        for i in range(self.N):
            res_gradient.append(self_gradient[i] + sign * sym_gradient[i])
            cur_idx += self.dimensions[i]
        return res_gradient




    def implicit_gradients(self, action_profile):
        '''
            the implicit gradient dynamics
            args:
                current action profile (a list of numpy arrays) of size N
            returns:
                the corresponding implicit gradient (a list of numpy arrays)
                 of size N
        '''

        applied_gradients = [np.zeros((1, self.dimensions[i])) for i in range(self.N)]

        #recording \nabla_{\phi_l}\Phi_{l+1}
        hessians_from_bottom = dict([(n, {}) for n in self.leaves[0]])
        for k in self.leaves[0]:
            #print(" k = ", k)
            #print("self leaves = ", self.leaves[0])
            hessians_from_bottom[k][k] = np.identity(self.dimensions[k])

        bottom_actions = [action_profile[i] for i in self.level_to_idxs[self.num_of_levels-1]]
        #print("bottom_actions = ", bottom_actions)
        for l in range(self.num_of_levels-1, -1, -1):
            for i in self.level_to_idxs[l]:
                self_action = action_profile[i]
                parent_action = action_profile[self.parents[i]]
                # one's partial gradient
                self_gradient = self.self_gradient_funcs[i](self_action, parent_action, bottom_actions)

                assert self_gradient.shape == (1, self.dimensions[i])

                applied_gradients[i] += self_gradient
                if(l == self.num_of_levels - 1):
                    continue

                # the set of \nabla_{a_{L, k}} u_{l, i}: {1 x d_{L, k}} vectors for all leaves k of i
                bottom_gradients = self.bottom_gradient_funcs[i](self_action, parent_action, bottom_actions)



                bottom_to_self_hessians = dict([(k, np.zeros((self.dimensions[k], self.dimensions[i]))) for k in self.leaves[i]])

                for j in self.children[i]:

                    # print("i = ", i)
                    # print("j = ", j)
                    # the hessian matrix propogated between consecutive levels
                    child_to_parent_hessian = self.child_to_parent_hessian_funcs[j](action_profile[j], action_profile[i], bottom_actions)
                    assert child_to_parent_hessian.shape == (self.dimensions[j], self.dimensions[i])

                    # only paths through direct children counts
                    for k in self.leaves[j]:
                        # (d_{L, k} x d_{l+1, j)) matrix multiply (d_{l+1, j)x d_{l, i)) matrix
                        #print("k = ", k)
                        #print("j = ", j)
                        assert hessians_from_bottom[k][j].shape == (self.dimensions[k], self.dimensions[j])

                        bottom_to_self_hessians[k] += hessians_from_bottom[k][j] @ child_to_parent_hessian

                for k in self.leaves[i]:
                    # only gradient pumped from leaves descendants accounts for the second term in total derivativese

                    #print("bottom_gradients[k] = ", bottom_gradients)
                    #print("bottom_to_self_hessians[k] = ", bottom_to_self_hessians[k])
                    assert bottom_gradients[k].shape == (1, self.dimensions[k])
                    applied_gradients[i] += bottom_gradients[k] @ bottom_to_self_hessians[k]

                    # recorded for upper-level usages
                    hessians_from_bottom[k][i] = bottom_to_self_hessians[k]


        return applied_gradients

    def second_order(self, action_profile, eps=0.000001):
        '''
            get total second-order for 1-d actions using finite difference approximation
        '''
        bottom_actions = [action_profile[i] for i in self.level_to_idxs[self.num_of_levels-1]]
        res = [np.zeros((1, 1)) for _ in range(self.N)]

        for l in range(self.num_of_levels-1, -1, -1):
            for i in self.level_to_idxs[l]:
                action_profile_1 = copy.deepcopy(action_profile)
                action_profile_2 = copy.deepcopy(action_profile)
                action_profile_1[i] += eps
                action_profile_2[i] -= eps
                queue = []
                queue.append((i, 1))
                while(len(queue)):
                    item = queue[0]
                    queue.pop(0)
                    if(item[0] in self.leaves[i]): break
                    for j in self.children[item[0]]:
                        inc = self.child_to_parent_hessian_funcs[j](action_profile[j], action_profile[i], bottom_actions) * item[1]
                        action_profile_1[j] += inc * eps
                        action_profile_2[j] -= inc * eps
                        if(j not in self.leaves[i]):
                            queue.append((j, inc))
                imp_grads_1 = self.implicit_gradients(action_profile_1)
                imp_grads_2 = self.implicit_gradients(action_profile_2)
                res[i] = (imp_grads_1[i]-imp_grads_2[i])/eps
        return res





    def get_payoffs(self, action_profile):
        '''
            get a list of payoff (costs)
            args:
                current action profile (a list of numpy arrays) of size N
            returns:
                the corresponding implicit gradient (a list of numpy arrays)
                 of size N
        '''

        payoff_list = [0.0 for i in range(self.N)]

        bottom_actions = [action_profile[i] for i in self.level_to_idxs[self.num_of_levels-1]]

        # for each level l
        for l in range(self.num_of_levels-1, -1, -1):
            # for agent i in level l
            for i in self.level_to_idxs[l]:
                self_action = action_profile[i]
                parent_action = action_profile[self.parents[i]]
                # one's payoff
                _self_payoff = self.payoff_funcs[i](self_action, parent_action, bottom_actions).item()
                payoff_list[i] += _self_payoff

        return payoff_list

