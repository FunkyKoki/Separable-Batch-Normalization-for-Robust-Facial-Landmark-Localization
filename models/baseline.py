import torch
import torch.nn as nn


class Baseline(nn.Module):
    
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2*2*2048, 2048)
        self.fc2 = nn.Linear(2048, 2*98)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.)
                
    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv6(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
    def calculateLoss(self, out, target):
        criterion = torch.nn.L1Loss()
        return criterion(out, target)
