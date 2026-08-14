import torch
import torch.nn as nn

class MyNeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Deep CNN Feature Extractor (VGG-style)
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global Average Pooling squashes any spatial dimension (H, W) down to 1x1.
        # This completely removes the need to calculate hardcoded flattened sizes!
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Light MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 1. Extract features (Output shape: [batch, 64, 8, 8])
        x = self.feature_extractor(x)
        
        # 2. Adaptive Pool (Output shape: [batch, 64, 1, 1])
        x = self.adaptive_pool(x)
        
        # 3. Flatten (Output shape: [batch, 64])
        x = torch.flatten(x, 1)
        
        # 4. Classify (Output shape: [batch, 10])
        return self.classifier(x)