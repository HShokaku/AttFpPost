from typing import List
from tap import Tap
import torch

class CommonArgs(Tap):

    no_cuda: bool       = False
    gpu: int            = None
    batch_size: int     = 512
    at_least_epoch: int = 0

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @property
    def cuda(self) -> bool:
        return not self.no_cuda and torch.cuda.is_available()
