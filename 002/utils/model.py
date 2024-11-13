import torch.nn as nn
from utils.initvars import  img_targetSize
class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1, padding=1):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class IntelModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(IntelModel,self).__init__()
        self.conv1 = convBlock(in_channels, 64)
        self.conv2 = convBlock(64, 128)
        self.conv3 = convBlock(128, 256)
        self.conv4 = convBlock(256, 512)
        self.flatten = nn.Flatten()
        multiplier = img_targetSize[0]//(2**4) # 2**4 maxpool size to the power of the number of maxpool layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*multiplier*multiplier, 1024)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x