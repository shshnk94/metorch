import torch
import torch.nn as nn
import torch.nn.functional as F
#model - pick from Srivastava et.al
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 96, 5, 1, 2)
        self.conv2 = nn.Conv2d(96, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(128, 256, 5, 1, 2)

        self.pool = nn.MaxPool2d(3, 2)
        
        self.fc1 = nn.Linear(2304, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
