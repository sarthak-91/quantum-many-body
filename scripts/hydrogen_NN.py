from torch.func import hessian,vmap
from quantum_lib.network import laplacian
import torch 
import torch.nn as nn 
from quantum_lib.monte_carlo import monte_carlo_sample

def update_lr(optimizer, std_value, proportional=0.5, min_lr=1e-3):
    new_lr = max(min_lr, proportional * std_value.item())  
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr  
    return new_lr

class exp_wave(nn.Module):
    def __init__(self):
        super().__init__() 
        self.a = nn.Parameter(data=torch.rand(()))
    def forward(self,r):
        return r * torch.exp(- self.a * r)
    def string(self):
        return 'psi = r * exp(- {:.3f} * r)'.format(self.a.item())


def e_loc(wave_function, r, l=0, batch_dim=True):
    l_eigen = l*(l+1)
    wave_func_values = wave_function(r)
    w_ = -0.5 * laplacian(wave_function, r, batch_dim=batch_dim)
    return w_/wave_func_values - 1/r + l_eigen/(2*r**2)

step_size = 0.1
min_value = 1e-5
wave_func = exp_wave()
optimizer = torch.optim.Adam(wave_func.parameters(), lr=0.1)
probab = lambda r: wave_func(r) ** 2
update_rule = lambda x: torch.clamp(x + torch.randn(1) * step_size, min_value, None)
domain_check = lambda r: r.item() > 0  

for i in range(200):
    r = torch.tensor([1.0])
    list_of_samples = monte_carlo_sample(input_vector=r,probab=probab, domain_check=domain_check, 
                                         update_rule=update_rule,
                                         n_samples=100000,sample_gap=30, burn_in=2000)  
    wave_func_tensor = wave_func(list_of_samples)
    energy_values = e_loc(wave_function=wave_func, r=list_of_samples,batch_dim=True)
    expect_value = torch.mean(energy_values)
    std_value =  torch.std(energy_values)
    loss = expect_value + std_value
    new_lr = update_lr(optimizer, std_value)
    print("i:{},N:{},E:{:.5f}, std:{:.5f}".format(i+1, len(list_of_samples), 
                                                  expect_value.item(), 
                                                  std_value.item()))
    print(wave_func.string(),"new_lr = {:.3f}".format(new_lr))
    if std_value < 1e-3:
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
