from quantum_lib.monte_carlo import monte_carlo_sample 
from quantum_lib.network import gradient, NN 
import torch 
import torch.nn as nn 

"""Harmoni Oscillator"""

class exp_wave(nn.Module):
    def __init__(self):
        super().__init__() 
        self.a = nn.Parameter(data=torch.randn(()))
        self.b = nn.Parameter(data=torch.randn(()))
        self.c = nn.Parameter(data=torch.randn(()))
    def forward(self,y):
        return (self.a + self.b*y + self.c*y**2)  * torch.exp(- y * y/2)
    def string(self):
        return 'psi = {:.3f} + {:.3f}*y + {:.3f}*y^2)* exp(-r*r/2)'.format(self.a.item(), self.b.item(),self.c.item())


def e_loc(wave_function, y):
    return -0.5*gradient(wave_function, y,2)/wave_function + 0.5*y**2


if __name__ == "__main__":
    step_size = 0.1
    wave_func = exp_wave()
    optimizer = torch.optim.Adam(wave_func.parameters(), lr=0.1)
    probab = lambda r: wave_func(r) ** 2
    update_rule = lambda x:x + torch.randn(1) * step_size

    for i in range(200):
        r = torch.tensor([1.0],requires_grad=True)
        list_of_samples = monte_carlo_sample(input_vector=r,probab=probab, update_rule=update_rule,
                                             n_samples=100000,sample_gap=30, burn_in=2000)  
        wave_func_tensor = wave_func(list_of_samples)
        energy_values = e_loc(y=list_of_samples,wave_function= wave_func_tensor)
        expect_value = torch.mean(energy_values)
        std_value =  torch.std(energy_values)
        loss = expect_value + std_value

        print("i:{},N:{},E:{:.5f}, std:{:.5f}".format(i+1, len(list_of_samples), 
                                                      expect_value.item(), 
                                                      std_value.item()))
        print(wave_func.string())
        if std_value < 1e-3:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
