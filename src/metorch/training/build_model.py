import torch.nn as nn

from ..models import Net


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
