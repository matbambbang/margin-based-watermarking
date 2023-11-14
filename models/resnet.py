import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# reference: https://github.com/huawei-noah/Efficient-Computing/blob/master/Data-Efficient-Model-Compression/DFND/resnet.py
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.out_feature = False
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, (1,1))
        # print(out.shape)
        feature = out.view(out.size(0), -1)
        # print(feature.shape)
        out = self.linear(feature)
        if self.out_feature == False:
            return out
        else:
            return out,feature
 
 
def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
 
def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)
 
def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)


class StandardNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mu', torch.Tensor((0.5, 0.5, 0.5)).view(3,1,1))
        self.register_buffer('std', torch.Tensor((0.5, 0.5, 0.5)).view(3,1,1))

    def forward(self, x):
        return (x - self.mu) / self.std

class CIFAR10Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mu', torch.Tensor((0.4914, 0.4822, 0.4465)).view(3,1,1))
        self.register_buffer('std', torch.Tensor((0.2023, 0.1994, 0.2010)).view(3,1,1))

    def forward(self, x):
        return (x - self.mu) / self.std

class CIFAR100Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mu', torch.Tensor((0.5071, 0.4867, 0.4408)).view(3,1,1))
        self.register_buffer('std', torch.Tensor((0.2675, 0.2565, 0.2761)).view(3,1,1))

    def forward(self, x):
        return (x - self.mu) / self.std
    
    
def general_res18(num_classes=10):
    return nn.Sequential(
        StandardNormalize(),
        resnet18(num_classes=num_classes))
    
def general_res34(num_classes=10):
    return nn.Sequential(
        StandardNormalize(),
        resnet34(num_classes=num_classes))
    
def general_res50(num_classes=10):
    return nn.Sequential(
        StandardNormalize(),
        resnet50(num_classes=num_classes))
    
def general_res101(num_classes=10):
    return nn.Sequential(
        StandardNormalize(),
        resnet101(num_classes=num_classes))
    
def general_res152(num_classes=10):
    return nn.Sequential(
        StandardNormalize(),
        resnet152(num_classes=num_classes))
    

# 'res18', 'res34', 'res50', 'res101', 'res152'
models = {
    'res18': general_res18,
    'res34': general_res34,
    'res50': general_res50,
    'res101': general_res101,
    'res152': general_res152
}