import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from ray import tune

from .utils import get_dataloaders, max_norm
from .models import Net
from .test import test

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps')

MODEL_DICT = {'dropout_net': Net}
#init here and put into GPU/CPU - later write build_model
def build_model(config):
    
    model = MODEL_DICT[config['model']]().to(device)
    for name, params in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(params)
        else:
            nn.init.constant_(params, 0)
            
    return model

#training loop
def train(config):
    
    train_loader, valid_loader = get_dataloaders(config)
    model = build_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    #checking init_loss - add an if condition here.
    print('Init loss: ', np.mean([criterion(model(x.to(device)), y.to(device)).item() for x, y in train_loader]))

    for epoch in range(config['epochs']):
    
        model.train()
        running_loss = 0.0
    
        # for name, params in model.named_parameters():
        #     if 'weight' in name and 'fc3' in name:
        #         print(name, params.mean().item(), params.std().item())
            
        for x, y in train_loader:
        
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
        
            logits = model(x)
            loss = criterion(logits, y)
        
            loss.backward()    
            optimizer.step()
        
            max_norm(model, config['max_norm_constraint'])
            running_loss += loss.item()
        
        training_loss = running_loss / len(train_loader)
    
        validation_loss, validation_metric = test(model, criterion, valid_loader)
        # validation_loss, validation_metric = 0.0, 0.0
    
        tune.report(training_loss=training_loss, validation_loss=validation_loss, validation_metric=validation_metric)
        print(f"Epoch: {epoch} Training loss: {training_loss}  Validation loss: {validation_loss} Validation metric: {validation_metric}")