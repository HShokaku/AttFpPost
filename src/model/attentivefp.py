from rdkit import Chem
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from src.model.base import ModelBase
from src.config.attentivefp import attentivefpArgs
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from src.utils import initialize_weights, get_activation_function
from src.utils.model.metrics import get_metric_func
from tqdm import tqdm
import pandas as pd

class attentivefp(ModelBase):

    def __init__(self, args=attentivefpArgs().parse_args([], known_only=True), save_dir=None):
        super(attentivefp, self).__init__(args, save_dir=save_dir)

    def _build(self, args):
        self.dataset_type   =  args.dataset_type
        self.ffn_dropout    =  nn.Dropout(args.dropout)
        self.mve            =  args.mve
        self.device         =  args.device
        self.num_tasks      =  args.num_tasks
        self.output_size    =  args.num_tasks
        self.loss_func      =  args.loss_func
        self.train_mean     =  args.train_mean
        self.train_std      =  args.train_std

        if self.dataset_type == "multiclass":
            self.output_size *= args.multiclass_num_classes
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.dataset_type == "classification":
            self.sigmoid = nn.Sigmoid()

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)
        self.to(self.device)

    @property
    def _model_type(self):
        return "attentivefp"

    def _forward(self, inputs):

        _output = self.ffn(self.encoder(inputs.to(self.device),
                                        inputs.ndata["atomic"].to(self.device),
                                        inputs.edata["type"].to(self.device)))


        if self.mve:
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)

            # mve can be true only in the case of regression task
            if self.training:
                return output, logvar
            else:
                output   = output * self.train_std + self.train_mean
                variance = torch.exp(logvar) * (self.train_std ** 2)
                return output, variance
        else:
            if self.training:
                output = self.output_layer(_output)
            else:
                output = self.output_layer(_output) * self.train_std + self.train_mean

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.dataset_type == "classification" and not self.training:
            assert self.train_std  == 1
            assert self.train_mean == 0
            output = self.sigmoid(output)

        if self.dataset_type == "multiclass":
            assert self.train_std  == 1
            assert self.train_mean == 0
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output, None

    def _loss(self,
              outputs,
              targets):
        mask    = torch.Tensor([[x is not None for x in tb] for tb in targets])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets])
        targets = (targets-self.train_mean) / self.train_std

        mask    = mask.to(self.device)
        targets = targets.to(self.device)

        if self.mve:
            loss = self.loss_func(targets, outputs[0], outputs[1]) * mask
        else:
            loss = self.loss_func(outputs[0], targets) * mask

        return loss.sum() / mask.sum()

    def predict_on_dataloader(self, dataloader):
        preds    = []
        variance = [] if self.mve else None
        targets  = []

        for batch in tqdm(dataloader):
            batch_preds, batch_var = self.predict_on_batch(batch[0])

            preds.extend(batch_preds.data.cpu().numpy().tolist())
            targets.extend(batch[1].data.cpu().numpy().tolist())
            if self.mve:
                variance.extend(batch_var.data.cpu().numpy().tolist())

        return preds, variance, targets

    def report_on_dataloader(self, dataloader, path):
        preds, variance, targets = self.predict_on_dataloader(dataloader)
        preds = np.array(preds)
        if variance is not None:
            variance = np.array(variance)
        smiles = dataloader.smiles

        output_dict = {}
        for i in range(self._config.number_of_molecules):
            output_dict[f"smiles_{i}"] = [s[i] for s in smiles]

        targets = np.array(targets)
        for i, name in enumerate(self._config.task_names):
            output_dict[name+"_label"] = targets[:, i]
            output_dict[name+"_pred"]  = preds[:, i]
            if variance is not None:
                output_dict[name+'_var'] = variance[:, i]

        df = pd.DataFrame(output_dict)
        df.to_csv(os.path.join(self.save_dir, path), index=False)

        file_name = path.split(".")[0]
        metrics   = self.eval_on_dataloader(dataloader)
        df        = pd.DataFrame(metrics)
        df.to_csv(os.path.join(self.save_dir, path.replace(file_name, file_name+"_performance")), index=False)

    def eval_on_dataloader(self, dataloader):
        preds, variance, targets = self.predict_on_dataloader(dataloader)
        metric_to_func = {metric: get_metric_func(metric) for metric in self._config.metrics}

        valid_preds   = [[] for _ in range(self.num_tasks)]
        valid_targets = [[] for _ in range(self.num_tasks)]
        for i in range(self.num_tasks):
            for j in range(len(preds)):
                if targets[j][i] is not None:  # Skip those without targets
                    valid_preds[i].append(preds[j][i])
                    valid_targets[i].append(targets[j][i])

        results = defaultdict(list)
        for i in range(self.num_tasks):
            for metric, metric_func in metric_to_func.items():
                results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

        results = dict(results)
        return results

    def create_encoder(self, args : attentivefpArgs) -> None:
        self.encoder = Encoder(args)

    def create_ffn(self, args : attentivefpArgs) -> None:

        first_linear_dim = args.hidden_size
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = []
            last_linear_dim = first_linear_dim
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
            ])
            last_linear_dim = args.ffn_hidden_size

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

        if self.mve:
            self.output_layer = nn.Linear(last_linear_dim, self.output_size)
            self.logvar_layer = nn.Linear(last_linear_dim, self.output_size)
        else:
            self.output_layer = nn.Linear(last_linear_dim, self.output_size)

    def featurize(self, inputs) -> torch.FloatTensor:
        return self.ffn[:-2](self.encoder(inputs))

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