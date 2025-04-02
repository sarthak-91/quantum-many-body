import numpy as np 
import torch
import matplotlib.pyplot as plt
from quantum_lib.monte_carlo import monte_carlo_sample
from quantum_lib.network import gradient
"""Quatum monte carlo for Hydrogen Atom
"""

def t_wave(r:torch.Tensor,alpha:float=1):
    return r * torch.exp(-alpha*r)



def hamiltonian(wave_func:torch.Tensor,r:torch.Tensor,l=0):
    l_eigen = l*(l+1)
    w__ = -0.5 * gradient(wave_func, r, 2)
    return w__ - wave_func/r + l_eigen*wave_func/(2 * r**2)

def e_loc(r:torch.Tensor,wave_func,l=0):
    return hamiltonian(wave_func=wave_func, r=r, l=l)/wave_func


step_size = 0.2
min_value = 1e-5

def update_rule(x):
    return torch.clamp(x + torch.randn(1) * step_size, min_value, None)

def domain_check(x):
    return  x.item() > 0  


    

if __name__ == "__main__":

    for i in [0.7,8/9,1,1.1]:
        r = torch.tensor([1.0],requires_grad=True) #starting vector 

        def probab(r):
            return t_wave(r, i) ** 2
    
        list_of_samples = monte_carlo_sample(input_vector=r,probab=probab, 
                                               domain_check=domain_check, 
                                               update_rule=update_rule,
                                               n_samples=90000,sample_gap=50, burn_in=1000) 


        wave_func_tensor = t_wave(list_of_samples,alpha=i)
        energy_values = e_loc(list_of_samples,wave_func= wave_func_tensor )
        expect_value = torch.mean(energy_values)
        std_value =  torch.std(energy_values)
        print("{:.2f} Num of samples = {} Expectation: E= {:.12f}, std={:.12f}".format(i,len(list_of_samples),
                                                                                         expect_value, std_value))
