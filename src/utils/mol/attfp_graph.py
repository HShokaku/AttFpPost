import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import dgllife
from dgllife.utils import smiles_to_bigraph

import torch
import pandas as pd
from src.utils.mol.feature_atom import atom_features
from src.utils.mol.feature_bond import bond_features
from rdkit import Chem
from torch.utils.data import Sampler
from tqdm import tqdm
from random import Random
import numpy as np
from typing import Iterator
import threading

def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(list(atom_features(atom)))
    return {'atomic': torch.tensor(feats).float()}

def featurize_edges(mol):
    feats     = []
    for bond in mol.GetBonds():
        feats.append(list(bond_features(bond)))
        feats.append(list(bond_features(bond)))
    return {'type': torch.tensor(feats).float()}


def collate_fn(samples):
    graph_list = [s[0] for s in samples]
    label_list = [[s[1]] for s in samples]

    return dgl.batch(graph_list), \
           torch.tensor(label_list).long()


class MoleculeDataset(DGLDataset):
    def __init__(self, df_path):
        self.df_path = df_path
        self.graph_list = []
        self.labels = []
        self.smiles_list = []
        super(MoleculeDataset, self).__init__(name='molecule')

    def process(self):
        df = pd.read_csv(self.df_path)
        for row in tqdm(df.iterrows()):
            smiles = row[1]["smiles"]
            label = row[1]["Y"]
            self.smiles_list.append(smiles)
            self.graph_list.append(smiles_to_bigraph(smiles,
                                                     node_featurizer=featurize_atoms,
                                                     edge_featurizer=featurize_edges))

            self.labels.append(label)

    def __getitem__(self, idx):
        return self.graph_list[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class MoleculeSampler(Sampler):

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):

        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle
        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([a[1] == 1 for a in dataset])
            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()
            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None
            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)
            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                self._random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.length

class MoleculeDataLoader(GraphDataLoader):

    def __init__(self,
                 dataset       : MoleculeDataset,
                 batch_size    : int = 50,
                 num_workers   : int = 0,
                 class_balance : bool = False,
                 shuffle       : bool = False,
                 seed          : int = 0):

        self._dataset       = dataset
        self._batch_size    = batch_size
        self._num_workers   = num_workers
        self._class_balance = class_balance
        self._shuffle       = shuffle
        self._seed          = seed
        self._context       = None
        self._timeout       = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset       = self._dataset,
            class_balance = self._class_balance,
            shuffle       = self._shuffle,
            seed          = self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset                 = self._dataset,
            batch_size              = self._batch_size,
            sampler                 = self._sampler,
            num_workers             = self._num_workers,
            collate_fn              = collate_fn,
            multiprocessing_context = self._context,
            timeout                 = self._timeout
        )

    @property
    def iter_size(self) -> int:
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        return super(MoleculeDataLoader, self).__iter__()
