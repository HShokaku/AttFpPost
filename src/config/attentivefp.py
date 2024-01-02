from tempfile import TemporaryDirectory
from typing import List
from typing_extensions import Literal
from src.utils.mol.dataset import set_cache_mol, empty_cache
from src.utils.model.loss import heteroscedastic_loss
import torch
import platform
from src.config.base import CommonArgs

'''
This section of code was adapted from the Chemprop project
Original Chemprop code is under the MIT License:
https://github.com/chemprop/chemprop/blob/d2b243939f12e22b3a1d0a4b2d3599852975cf2b/chemprop/args.py
'''

Metric = Literal['roc-auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy', 'EF1']

class attentivefpArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""
    number_of_molecules    : int = 1
    atom_descriptors       : Literal['feature', 'descriptor'] = None
    atom_descriptors_size  : int  = None
    no_cache_mol           : bool = False
    empty_cache            : bool = False
    task_names             : List[str] = ["property"]
    dataset_type           : Literal['regression', 'classification', 'multiclass'] = 'regression'
    multiclass_num_classes : int = 3
    pytorch_seed           : int = 0
    metric                 : Metric = None
    extra_metrics          : List[Metric] = []
    quiet                  : bool = False
    log_frequency          : int = 1
    bias                   : bool = False
    depth                  : int = 3
    mpn_shared             : bool = False
    dropout                : float = 0.0
    activation             : Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    atom_messages          : bool = False
    undirected             : bool = False
    features_size          : int = None
    hidden_size            : int = 300
    ffn_hidden_size        : int = None
    ffn_num_layers         : int = 4
    features_only          : bool = False
    aggregation            : Literal['mean', 'sum', 'norm'] = 'mean'
    aggregation_norm       : int = 100
    reaction               : bool = False
    reaction_mode          : Literal['reac_prod', 'reac_diff', 'prod_diff'] = 'reac_diff'
    explicit_h             : bool = False

    init_lr                : float = 1e-4
    grad_clip              : float = None
    mve                    : bool = False
    early_stopping_num     : int  = 50
    train_mean             : int = 0
    train_std              : int = 1

    input_feature_dim      : int = 200
    fingerprint_dim        : int = 200
    radius                 : int = 2
    p_dropout              : float = 0.03
    T                      : int = 1


    def __init__(self, *args, **kwargs) -> None:
        super(attentivefpArgs, self).__init__(*args, **kwargs)
        self._num_tasks = None
        self._features_size = None

    @property
    def metrics(self) -> List[str]:
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy'}

    @property
    def use_input_features(self) -> bool:
        return self.features_size is not None

    @property
    def num_tasks(self) -> int:
        return len(self.task_names) if self.task_names is not None else 1

    @property
    def loss_func(self):
        if self.mve:
            return heteroscedastic_loss
        if self.dataset_type == "regression":
            return torch.nn.MSELoss(reduction='none')
        elif self.dataset_type == "classification":
            return torch.nn.BCEWithLogitsLoss(reduction='none')

    def process_args(self) -> None:

        set_cache_mol(not self.no_cache_mol)

        if self.empty_cache:
            empty_cache()

        global temp_dir  # Prevents the temporary directory from being deleted upon function return


        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'roc-auc'
            elif self.dataset_type == 'multiclass':
                self.metric = 'cross_entropy'
            else:
                self.metric = 'rmse'

        if self.metric in self.extra_metrics:
            raise ValueError(f'Metric {self.metric} is both the metric and is in extra_metrics. '
                             f'Please only include it once.')

        for metric in self.metrics:
            if not ((self.dataset_type == 'classification' and metric in ['auc', 'prc-auc', 'accuracy', 'binary_cross_entropy']) or
                    (self.dataset_type == 'regression' and metric in ['rmse', 'mae', 'mse', 'r2']) or
                    (self.dataset_type == 'multiclass' and metric in ['cross_entropy', 'accuracy'])):
                raise ValueError(f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".')

        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        if self.mve and self.dataset_type != "regression":
            raise ValueError('For MVE mode, the dataset type must be "regression".')



