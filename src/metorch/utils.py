import torch
from torch.utils.data import random_split, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

#preparing the dataloaders
def get_dataloaders(config):
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if config['mode'] == 'train':
        data = CIFAR10('./data', download=True, transform=transform)
        train_size = int(0.8 * len(data))
        train_set, valid_set = random_split(
            data, 
            [train_size, len(data) - train_size], 
            generator=torch.Generator().manual_seed(42))

        #sample for debugging
        #train_set = Subset(train_set, np.arange(batch_size))
        #valid_set = Subset(valid_set, np.arange(batch_size))

        train_loader = DataLoader(train_set, batch_size=config['batch_size'])
        valid_loader = DataLoader(valid_set, batch_size=config['batch_size'])
        
        return train_loader, valid_loader

    else:
        test_set = CIFAR10('./data', download=True, transform=transform, train=False)
        test_loader = DataLoader(test_set, batch_size=config['batch_size'])
        
        return test_loader

def max_norm(model, max_val, eps=1e-8):
    
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))
