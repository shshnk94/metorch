import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

#training loop
def train(config):
    
    train_loader, valid_loader = get_dataloaders(config)
    model = build_model()
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
