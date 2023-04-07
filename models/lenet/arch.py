import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes):
        # (C, H, W)
        # input (1, 32, 32)
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5), # (6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (6, 14, 14)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5), # (16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (16, 5, 5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120), # (120)
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84), # (84)
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, num_classes) # (num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.flatten(1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out