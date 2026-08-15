import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, pool=False):
    """A helper function to build a Conv -> BatchNorm -> ReLU block"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        # pyrefly: ignore [bad-argument-type]
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class MyNeuralNet(nn.Module):
    """
    ResNet9 Architecture.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Prep layer
        self.prep = conv_block(3, 64)
        
        # Layer 1 (with Residual Block)
        self.layer1 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )
        
        # Layer 2
        self.layer2 = conv_block(128, 256, pool=True)
        
        # Layer 3 (with Residual Block)
        self.layer3 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.prep(x)
        
        x = self.layer1(x)
        # SKIP CONNECTION: Add the input of res1 directly to its output!
        x = self.res1(x) + x
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        # SKIP CONNECTION: Add the input of res2 directly to its output!
        x = self.res2(x) + x
        
        x = self.classifier(x)
        return x