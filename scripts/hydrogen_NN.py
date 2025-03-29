import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from quantum_lib.monte_carlo import monte_carlo_sample
from quantum_lib.network import gradient, NN
"""Quatum monte carlo for Hydrogen Atom
"""
class exp_wave(nn.Module):
    def __init__(self):
        super().__init__() 
        self.a = nn.Parameter(data=torch.rand(()))
    def forward(self,r):
        return self.a * r * torch.exp(- self.a * r)
    def string(self):
        return f'psi = {self.a.item()} * r * exp(- {self.a.item()} * r)'

def hamiltonian(wave_func:torch.Tensor,r:torch.Tensor,l=0):
    l_eigen = l*(l+1)
    w__ = -0.5 * gradient(wave_func, r, 2)
    return w__ - wave_func/r + l_eigen*wave_func/(2 * r**2)

def e_loc(r:torch.Tensor,wave_func,l=0):
    return hamiltonian(wave_func=wave_func, r=r, l=l)/wave_func


step_size = 0.1
min_value = 1e-5
wave_func = exp_wave()
optimizer = torch.optim.Adam(wave_func.parameters(), lr=0.1)
probab = lambda r: wave_func(r) ** 2
update_rule = lambda x: torch.clamp(x + torch.randn(1) * step_size, min_value, None)
domain_check = lambda r: r.item() > 0  

for i in range(200):
    r = torch.tensor([1.0],requires_grad=True)
    list_of_samples = monte_carlo_sample(input_vector=r,probab=probab, domain_check=domain_check, 
                                         update_rule=update_rule,
                                         n_samples=100000,sample_gap=30, burn_in=2000)  
    wave_func_tensor = wave_func(list_of_samples)
    energy_values = e_loc(list_of_samples,wave_func= wave_func_tensor)
    expect_value = torch.mean(energy_values)
    std_value =  torch.std(energy_values)
    loss = expect_value + std_value

    print("i:{},N:{},E:{:.5f}, std:{:.5f}".format(i+1, len(list_of_samples), 
                                                  expect_value.item(), 
                                                  std_value.item()))
    if std_value < 1e-3:
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
