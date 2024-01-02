from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
import torch.distributions as tdist
import torch.nn as nn
import torch

'''
The following code is adapted from the Posterior Network
Original Posterior Network code is licensed under the Hippocratic License:
https://github.com/sharpenb/Posterior-Network/blob/18c49e5fe164203cf8c9d69beafe808990dc92bc/src/posterior_networks/NormalizingFlowDensity.py
'''

class NormalizingFlowDensity(nn.Module):

    def __init__(self, dim, flow_length, flow_type='radial_flow', device=None):
        super(NormalizingFlowDensity, self).__init__()
        self.dim         = dim
        self.flow_length = flow_length
        self.flow_type   = flow_type
        self.device = device
        # self.mu = torch.zeros(self.dim)
        # self.cov = torch.eye(self.dim)


        if self.device is not None:
            self.mean = nn.Parameter(torch.zeros(self.dim).to(device), requires_grad=False)
            self.cov = nn.Parameter(torch.eye(self.dim).to(device), requires_grad=False)

            # self.mean = torch.zeros(self.dim).to(device)
            # self.cov = torch.eye(self.dim).to(device)
        else:
            self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
            self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        if self.flow_type == 'radial_flow':
            self.transforms = nn.Sequential(*(
                Radial(dim) for _ in range(flow_length)
            ))
        elif self.flow_type == 'iaf_flow':
            self.transforms = nn.Sequential(*(
                affine_autoregressive(dim, hidden_dims=[128, 128]) for _ in range(flow_length)
            ))
        else:
            raise NotImplementedError

    def forward(self, z):
        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians
        return log_prob_x
