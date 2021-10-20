import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


class DynamicSepBatchNorm2d(nn.Module):
    
    def __init__(self, inplanes, sep_ways=3, ratios=0.1, temp=30):
        super(DynamicSepBatchNorm2d, self).__init__()
        bn = []
        for i in range(sep_ways):
            bn.append(nn.BatchNorm2d(inplanes))
        self.bn = nn.ModuleList(bn)
        self.sep_ways = sep_ways
        self.inplanes = inplanes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(inplanes, int(inplanes*ratios), 1, bias=False)
        self.fc2 = nn.Conv2d(int(inplanes*ratios), sep_ways, 1, bias=False)
        self.temp = temp

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    
    def updata_temperature(self):
        if self.temp != 1:
            self.temp -= 1
            print('Change temperature to:', str(self.temp))
    
    def forward(self, x):
        batchsize, channel, height, width = x.size()
        out = self.avgpool(x)
        out = self.fc1(out)
        out = F.leaky_relu(out)
        out = self.fc2(out).view(out.size(0), -1)
        out = F.softmax(out/self.temp, 1)
        out_record = copy.deepcopy(out.cpu().detach().numpy())

        container = []
        for i in range(self.sep_ways):
            container.append(self.bn[i](x).unsqueeze(0))
        container = torch.cat(container)
        container = container.permute(1,0,2,3,4).contiguous().reshape(-1, channel, height, width)
        out = container * out.reshape(-1,).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return torch.sum(torch.reshape(out, (batchsize, self.sep_ways, channel, height, width)), dim=1), out_record


class BaselineSepDyBN(nn.Module):
    
    def __init__(self, sep_ways=3, ratios=0.1, temp=30):
        super(BaselineSepDyBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = DynamicSepBatchNorm2d(64,   sep_ways, ratios, temp)
        self.bn2 = DynamicSepBatchNorm2d(128,  sep_ways, ratios, temp)
        self.bn3 = DynamicSepBatchNorm2d(256,  sep_ways, ratios, temp)
        self.bn4 = DynamicSepBatchNorm2d(512,  sep_ways, ratios, temp)
        self.bn5 = DynamicSepBatchNorm2d(1024, sep_ways, ratios, temp)
        self.bn6 = DynamicSepBatchNorm2d(2048, sep_ways, ratios, temp)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2*2*2048, 2048)
        self.fc2 = nn.Linear(2048, 2*98)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def updata_temperature(self):
        for i in range(6):
            eval('self.bn' + str(i+1)).updata_temperature()
                
    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        out = self.conv1(x)
        out, out_record_bn1 = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out, out_record_bn2 = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out, out_record_bn3 = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out, out_record_bn4 = self.bn4(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        out, out_record_bn5 = self.bn5(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv6(out)
        out, out_record_bn6 = self.bn6(out)
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

