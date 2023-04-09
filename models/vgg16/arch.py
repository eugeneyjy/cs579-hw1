import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes, channels=3):
        # (N, C, H, W) -- (Batch, Channel, Height, Width)
        # input (N, C, 224, 224)
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1), # (N, 64, 224, 224)
            nn.ReLU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), # (N, 64, 224, 224)
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2) # (N, 64, 112, 112)

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), # (N, 128, 112, 112)
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), # (N, 128, 112, 112)
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2) # (N, 128, 56, 56)

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), # (N, 256, 56, 56)
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), # (N, 256, 56, 56)
            nn.ReLU()
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0), # (N, 256, 56, 56)
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2) # (N, 256, 28, 28)

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), # (N, 512, 28, 28)
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # (N, 512, 28, 28)
            nn.ReLU()
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0), # (N, 512, 28, 28)
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, 2) # (N, 512, 14, 14)

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # (N, 512, 14, 14)
            nn.ReLU()
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # (N, 512, 14, 14)
            nn.ReLU()
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0), # (N, 512, 14, 14)
            nn.ReLU()
        )
        self.pool5 = nn.MaxPool2d(2, 2) # (N, 512, 7, 7)

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # (N, 4096)
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096) # (N, 4096)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes) # (N, num_classes)
        )


    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.pool1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.pool2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.pool3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.pool4(out)

        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        out = self.pool5(out)

        out = out.flatten(1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out