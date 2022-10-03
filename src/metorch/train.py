import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from ray.air import session
from ray.train.torch import prepare_model
from ray.air.checkpoint import Checkpoint

from .utils import get_dataloaders, max_norm
from .models import DropoutNet
from .test import test

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DICT = {'dropout_net': DropoutNet}

#init here and put into GPU/CPU - later write build_model
def build_model(config):
    
    model = MODEL_DICT[config['model']](config).to(device)
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

    if not config['use_ray']:
        progress = open(os.path.join(config['results_dir'], 'progress.csv'), 'w')
        progress.write('Epoch, Training Loss, Validation Loss, Validation metric\n')
        with torch.no_grad():
            print('Init loss: ', np.mean([criterion(model(x.to(device)), y.to(device)).item() for x, y in train_loader]))

    for epoch in range(config['epochs']):
    
        model.train()
        running_loss = 0.0
            
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
        
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad(True)
            loss.backward()    
            optimizer.step()
        
            max_norm(model, config['max_norm_constraint'])
            running_loss += loss.item()
        
        training_loss = running_loss / len(train_loader)
        validation_loss, validation_metric = test(model, criterion, valid_loader)

        if config['use_ray']:
            #checkpoint = Checkpoint.from_directory(config['results_dir'])
            session.report({'training_loss':training_loss, 'validation_loss':validation_loss, 'validation_metric':validation_metric})
        else:
            torch.save(model.state_dict(), os.path.join(config['results_dir'], 'model.pth'))
            progress.write(f"{epoch}, {training_loss}, {validation_loss}, {validation_metric}\n")
            print(f"Epoch: {epoch} Training loss: {training_loss}  Validation loss: {validation_loss} Validation metric: {validation_metric}")

