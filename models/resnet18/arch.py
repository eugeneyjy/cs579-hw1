import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(ResNet18, self).__init__()

        self.curr_in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxPool = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_layer(64, 2, 1)
        self.conv3_x = self._make_layer(128, 2, 2)
        self.conv4_x = self._make_layer(256, 2, 2)
        self.conv5_x = self._make_layer(512, 2, 2)
        self.avgPool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.curr_in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(BasicBlock(self.curr_in_channels, out_channels, stride, downsample))
        self.curr_in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.curr_in_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.maxPool(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgPool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out