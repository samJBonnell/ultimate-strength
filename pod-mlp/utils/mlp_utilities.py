import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        
        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.linears.extend([nn.Linear(layers_size, layers_size) for i in range(1, self.num_layers-1)])
        self.linears.append(nn.Linear(layers_size, output_size))
    
    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = torch.relu(self.linears[i](x))
        
        x = self.linears[-1](x)
        return x