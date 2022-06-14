
from hessian import hessian
import numpy as np
import torch

class Payoff:
    '''
        a class calculate the gradients of payoff_functions
        implemented for 1-dim case

        args: 
            payoff_func: a payoff function, implemented by pytroch
            self_action: 1-dim, the agent's action
            parent_action: 1-dim, the parent's actopm
            bottom_actions: a vector of botton agents' action, np.array

    '''
    def __init__(self, payoff_func, bottom_idx):
    
        self.payoff_func = payoff_func
        self.bottom_idx = bottom_idx # idx: agent number in bottom level, value: actual agent number 


        
    def set_actions(self,self_action, parent_action, bottom_actions):
        
        self.self_action = torch.tensor(self_action)
        self.parent_action = torch.tensor(parent_action)
        

        bottom_actions_tensor = [torch.tensor(bottom_actions[i]) for i in range(len(bottom_actions))]
        self.bottom_actions = torch.cat(bottom_actions_tensor, 0)
        self.self_action.requires_grad_(True)
        self.parent_action.requires_grad_(True)
        self.bottom_actions.requires_grad_(True)
        

    def get_payoff_func(self, self_action, parent_action, bottom_actions):
        self.set_actions(self_action, parent_action, bottom_actions)
        return self.payoff_func(self.self_action, self.parent_action, self.bottom_actions)
    
    def get_gradient_func(self, self_action, parent_action, bottom_actions):
        # return a vector of (1 , dimensions[n])

        payoff = self.get_payoff_func(self_action, parent_action, bottom_actions)
        payoff.backward()

        dimension = self_action.shape[1] #dimisition

        res = np.array(self.self_action.grad.item()).reshape(1, dimension)

        return res
    
    def get_bottom_grad_func(self, self_action, parent_action, bottom_actions):
        payoff = self.get_payoff_func(self_action, parent_action, bottom_actions)
        payoff.backward()
        bottom_grads = self.bottom_actions.grad.numpy()
        _dict = {self.bottom_idx[i]: bottom_grads[i].reshape(1, bottom_actions[i].shape[1]) for i in range(len(bottom_grads))}
        return _dict
    
    def get_child_to_parent_hessian_func(self, self_action, parent_action, bottom_actions):

        payoff = self.get_payoff_func(self_action, parent_action, bottom_actions)
        _payoff_value = payoff[0, 0]

        h = hessian(_payoff_value, [self.self_action, self.parent_action], create_graph=True, allow_unused=True)

        N = - h[0,1].item()
        D = h[0,0].item()


        res = N/D*np.identity(1)
        
        return res




