import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs