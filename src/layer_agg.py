import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregateLayer(nn.Module):
    def __init__(self, module_test=False):
        super(AggregateLayer, self).__init__()
        self.module_test = module_test

    def forward(self, pref, c):
        '''
        input:
            pref : (batch_size, n_assign, dim_node)
            c    : (batch_size, 1   , dim_node)
        output:
            u    : (batch_size, dim_node)
        '''
        _weights = torch.mul(pref, c) # (batch_size, n_assign, dim_node)
        weights = torch.sum(_weights, dim=-1, keepdim=True) # (batch_size, n_assign, 1)
        pref_weighted = torch.mul(weights, pref) # (batch_size, n_assign, dim_node)
        u = torch.sum(pref_weighted, dim=-2, keepdim=True) # (batch_size, 1, dim_node)

        if self.module_test:
            var_input = ['pref', 'c']
            var_inmed = ['_weights', 'weights', 'pref_weighted']
            var_ouput = ['u']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )

        return u
    

if __name__=='__main__':

    from utils import module_test_print

    Module_Test = AggregateLayer(module_test=True)
    pref = torch.arange(8, dtype = torch.float32).view((2,2,2))
    c = torch.arange(4, dtype = torch.float32).view((2,1,2))
        
    output = Module_Test(pref, c)





