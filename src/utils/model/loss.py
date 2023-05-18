import torch
from torch.distributions.dirichlet import Dirichlet

def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    loss = precision * (true - mean) ** 2 + log_var
    return loss

def UCE_loss(alpha, soft_output):
    alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, 2)
    entropy_reg = Dirichlet(alpha).entropy()
    UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha)))  - 1e-5 * torch.sum(entropy_reg)
    return UCE_loss

def CE_loss(soft_output_pred, soft_output):
    CE_loss = - torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))
    return CE_loss