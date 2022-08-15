from sklearn.metrics import f1_score
import torch

def test(model, criterion, loader):
    
    model.eval()
    running_loss, running_metric = 0.0, 0.0
    
    with torch.no_grad():
        for x, y in loader:
        
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            metric = f1_score(y.cpu(), torch.argmax(logits, 1).cpu(), average='macro')
            
            running_loss += loss.item()
            running_metric += metric
        
    return running_loss / len(loader), running_metric / len(loader)
