import torch
import torch.nn as nn

class MyNeuralNet(nn.Module):
    def __init__(self, layers, size_list):
        super().__init__()

        self.input_size, self.hidden_size, self.output_size = size_list

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        self.bns.append(nn.BatchNorm1d(self.hidden_size))

        # Hidden layers
        for _ in range(layers - 2):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.bns.append(nn.BatchNorm1d(self.hidden_size))

        # Output
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i in range(len(self.layers)):
            identity = x
    
            x = self.layers[i](x)
            x = self.bns[i](x)
            x = torch.relu(x)
    
            # Add skip connection (only if dimensions match)
            if x.shape == identity.shape:
                x = x + identity
    
        return self.out(x)