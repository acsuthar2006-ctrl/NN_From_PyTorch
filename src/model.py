import torch
import torch.nn as nn

class MyNeuralNet(nn.Module):
    def __init__(self, layers, size_list):
        super().__init__()
        self.input_size, self.hidden_size, self.output_size = size_list

        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        self.out = nn.Linear(self.hidden_size, self.output_size)

        for _ in range(layers - 2):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out(x)
