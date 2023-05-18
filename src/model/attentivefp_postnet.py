from rdkit import Chem
import os
import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
from src.model.base import ModelBase
from src.config.attentivefp_postnet import attentivefpPostNetArgs
import torch.nn.functional as F
from src.utils.model.weight import initialize_weights
from src.utils.model.activation import get_activation_function
from src.utils.model.metrics import get_metric_func
from src.utils.model.normalizingflowdensity import NormalizingFlowDensity
from tqdm import tqdm
import pandas as pd
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout


class attentivefpPostNet(ModelBase):

    def __init__(self, args=attentivefpPostNetArgs().parse_args([], known_only=True), save_dir=None):
        super(attentivefpPostNet, self).__init__(args, save_dir=save_dir)

    def _build(self, args):
        self.ffn_dropout = nn.Dropout(args.dropout)
        self.device = args.device
        self.loss_func = args.loss_func
        self.density_type = args.density_type
        self.n_density = args.n_density
        # self.N              =  torch.nn.Parameter(torch.ones_like(torch.tensor([1., 1.])).to(self.device))
        self.N = args.N.to(self.device)
        self.num_tasks = 1

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

        if self.density_type == 'radial_flow' or self.density_type == 'iaf_flow':
            self.density_estimation = nn.ModuleList(
                [NormalizingFlowDensity(dim=args.latent_dim,
                                        flow_length=self.n_density,
                                        flow_type=self.density_type,
                                        device=self.device) for c
                 in range(2)])

        self.batch_norm = nn.BatchNorm1d(num_features=args.latent_dim)
        self.to(self.device)

    @property
    def _model_type(self):
        return "attentivefp_postnet"

    def _forward(self, input):
        zk = self.ffn(self.encoder(input.to(self.device),
                                   input.ndata["atomic"].to(self.device),
                                   input.edata["type"].to(self.device)))
        zk = self.batch_norm(zk)
        batch_size = len(zk)

        log_q_zk = torch.zeros((batch_size, 2)).to(self.device)
        alpha = torch.zeros((batch_size, 2)).to(self.device)

        if isinstance(self.density_estimation, nn.ModuleList):
            for c in range(2):
                log_p = self.density_estimation[c].log_prob(zk)
                log_q_zk[:, c] = log_p
                alpha[:, c] = 1. + (self.N[c] * torch.exp(log_q_zk[:, c]))
        else:
            log_q_zk = self.density_estimation.log_prob(zk)
            alpha = 1. + (self.N[:, None] * torch.exp(log_q_zk)).permute(1, 0)
        pass

        soft_output_pred = torch.nn.functional.normalize(alpha, p=1)

        return alpha, soft_output_pred

    def _loss(self,
              outputs,
              targets):
        alpha, soft_output_pred = outputs
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets])
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets])
        targets = targets.to(self.device).type(torch.int64)
        mask = mask.to(self.device)
        targets_hot = torch.zeros(targets.shape[0], 2)
        targets_hot = targets_hot.to(self.device)
        targets_hot.scatter_(1, targets, 1)
        # loss = - torch.sum(targets_hot.squeeze() * torch.log(soft_output_pred))
        loss = self.loss_func(alpha, targets_hot)

        return loss / mask.sum()

    def predict_on_dataloader(self, dataloader):
        alphas = []
        preds = []
        targets = []
        for batch in tqdm(dataloader):
            batch_alphas, batch_preds = self.predict_on_batch(batch[0])
            batch_preds = batch_preds[:, 1].unsqueeze(-1)
            preds.extend(batch_preds.data.cpu().numpy().tolist())
            alphas.extend(batch_alphas.data.cpu().numpy().tolist())
            targets.extend(batch[1].data.cpu().numpy().tolist())
        
        return alphas, preds, targets

    def report_on_dataloader(self, dataloader, path):
        alphas, preds, targets = self.predict_on_dataloader(dataloader)
        preds = np.array(preds)
        alphas = np.array(alphas)
        smiles = dataloader.smiles

        output_dict = {}
        for i in range(self._config.number_of_molecules):
            output_dict[f"smiles_{i}"] = [s[i] for s in smiles]

        targets = np.array(targets)
        for i, name in enumerate(self._config.task_names):
            output_dict[name + "_label"] = targets[:, i]
            output_dict[name + "_pred"] = preds[:, i]
            output_dict[name + "_alpha0"] = alphas[:, 0]
            output_dict[name + "_alpha1"] = alphas[:, 1]

        df = pd.DataFrame(output_dict)
        df.to_csv(os.path.join(self.save_dir, path), index=False)

        file_name = path.split(".")[0]
        metrics = self.eval_on_dataloader(dataloader)
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(self.save_dir, path.replace(file_name, file_name + "_performance")), index=False)

    def eval_on_dataloader(self, dataloader):
        alphas, preds, targets = self.predict_on_dataloader(dataloader)

        metric_to_func = {metric: get_metric_func(metric) for metric in self._config.metrics}

        valid_preds = [[] for _ in range(self.num_tasks)]
        valid_alphas = [[] for _ in range(self.num_tasks)]
        valid_targets = [[] for _ in range(self.num_tasks)]

        for i in range(self.num_tasks):
            for j in range(len(preds)):
                if targets[j][i] is not None:  # Skip those without targets
                    valid_preds[i].append(preds[j][i])
                    valid_targets[i].append(targets[j][i])
                    valid_alphas[i].append(alphas[j])

        results = defaultdict(list)
        for i in range(self.num_tasks):
            for metric, metric_func in metric_to_func.items():
                if metric == "UCE" or metric == "temp":
                    results[metric].append(metric_func(valid_targets[i], valid_alphas[i]))
                else:
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

        results = dict(results)
        return results

    def create_encoder(self, args: attentivefpPostNetArgs) -> None:
        self.encoder = Encoder(args)

    def create_ffn(self, args: attentivefpPostNetArgs) -> None:
        first_linear_dim = args.hidden_size
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                nn.Linear(first_linear_dim, args.latent_dim)
            ]

        else:
            ffn = [
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    self.ffn_dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])

            ffn.extend([
                activation,
                self.ffn_dropout,
                nn.Linear(args.ffn_hidden_size, args.latent_dim)])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        radius     = args.radius
        hidden_dim = args.hidden_size
        T          = args.T
        dropout    = args.p_dropout
        self.attfp = AttentiveFPGNN(133, 14, radius, hidden_dim, dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=hidden_dim,
                                          num_timesteps=T,
                                          dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, g, h, e):
        h = F.relu(self.attfp(g, h, e))

        with g.local_scope():
            g.ndata['h'] = h

            hg = self.readout(g, g.ndata['h'], False)
            return hg