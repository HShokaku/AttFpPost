import torch.nn as nn

def initialize_weights(model: nn.Module) -> None:

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)