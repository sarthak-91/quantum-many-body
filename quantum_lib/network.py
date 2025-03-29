import numpy as np
import torch
import torch.nn as nn

def gradient(y: torch.Tensor, x: torch.Tensor, order:int =1)->torch.Tensor:
    derivative = y
    for i in range(order):
        derivative =torch.autograd.grad(
                derivative,x,
                torch.ones_like(x),
                create_graph=True,retain_graph=True)[0]
    return derivative



class SinAct(torch.nn.Module):
    """custom sine activation function"""
    @staticmethod
    def forward(input):
        return torch.sin(input)


class NN(torch.nn.Module):

    def __init__(self,hidden_layers:list,output_num:int = 1):
        super(NN,self).__init__()
        self.activation = SinAct()
        input_layer = torch.nn.Linear(1,hidden_layers[0])
        self.Hidden_Layers = nn.ModuleList()
        self.Hidden_Layers.append(input_layer)
        for i in range(0,len(hidden_layers)-1):
            layer = torch.nn.Linear(hidden_layers[i],hidden_layers[i+1])
            self.Hidden_Layers.append(layer)
        output_layer = nn.Linear(hidden_layers[-1],output_num)
        self.Hidden_Layers.append(output_layer)

    def forward(self, input_tensor):
        y_out = input_tensor
        for i in range(0,len(self.Hidden_Layers)-1):
            z_out = self.Hidden_Layers[i](y_out)
            y_out = self.activation(z_out)

        output_tensor = self.Hidden_Layers[-1](y_out)
        return output_tensor
