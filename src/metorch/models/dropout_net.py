import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.drop_in= nn.Dropout(config['drop_in'])
        self.drop_conv = nn.Dropout(config['drop_conv'])
        self.drop_fc = nn.Dropout(config['drop_fc'])

        self.conv1 = nn.Conv2d(3, 96, 5, 1, 2)
        self.conv2 = nn.Conv2d(96, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(128, 256, 5, 1, 2)

        self.pool = nn.MaxPool2d(3, 2)
        
        self.fc1 = nn.Linear(2304, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(self.drop_in(x))))
        x = self.pool(F.relu(self.conv2(self.drop_conv(x))))
        x = self.pool(F.relu(self.conv3(self.drop_conv(x))))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(self.drop_fc(x)))
        x = F.relu(self.fc2(self.drop_fc(x)))
        x = self.fc3(self.drop_fc(x))
        
        return x
