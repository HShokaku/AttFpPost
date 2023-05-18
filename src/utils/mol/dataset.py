from typing import Dict, List, Optional, Union
from collections import OrderedDict
from torch.utils.data import Dataset
from rdkit import Chem
from random import Random
from src.utils.mol.feature_generator import get_features_generator
from src.utils.mol.feature_graph import MolGraph, BatchMolGraph
from src.utils import StandardScaler
import numpy as np

CACHE_GRAPH = False
CACHE_MOL = False
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}

def cache_graph() -> bool:
    return CACHE_GRAPH

def cache_mol() -> bool:
    return CACHE_MOL

def set_cache_graph(cache_graph: bool) -> None:
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph

def set_cache_mol(cache_mol: bool) -> None:
    global CACHE_MOL
    CACHE_MOL = cache_mol

def empty_cache():
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()

class MoleculeDatapoint:

    def __init__(self,
                 smiles:               List[str],
                 targets:              List[Optional[float]]  = None,
                 row:                  OrderedDict            = None,
                 features:             np.ndarray             = None,
                 features_generator:   List[str]              = None,
                 atom_features:        np.ndarray             = None,
                 atom_descriptors:     np.ndarray             = None):

        if features is not None and features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        self.smiles              =  smiles
        self.targets             =  targets
        self.row                 =  row
        self.features            =  features
        self.features_generator  =  features_generator
        self.atom_descriptors    =  atom_descriptors
        self.atom_features       =  atom_features

        if self.features_generator is not None:
            self.features = []
            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m in self.mol:
                    self.features.extend(features_generator(m))
            self.features = np.array(self.features)

        replace_token = 0
        if self.features         is not None:
            self.features         = np.where(np.isnan(self.features),         replace_token, self.features)
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)
        if self.atom_features    is not None:
            self.atom_features    = np.where(np.isnan(self.atom_features),    replace_token, self.atom_features)

        self.raw_features, self.raw_targets = self.features, self.targets

    @property
    def mol(self) -> List[Chem.Mol]:
        mol = [SMILES_TO_MOL.get(s, Chem.MolFromSmiles(s)) for s in self.smiles]
        assert None not in mol
        if cache_mol():                                                                                                  # cache_mol()
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m
        return mol

    @property
    def number_of_molecules(self) -> int:
        return len(self.smiles)

    def set_features(self, features: np.ndarray) -> None:
        self.features = features

    def extend_features(self, features: np.ndarray) -> None:
        assert len(features.shape) == 1
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        self.features, self.targets = self.raw_features, self.raw_targets

class MoleculeDataset(Dataset):

    def __init__(self, data: List[MoleculeDatapoint]):

        self._data = data
        self._scaler = None
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]
        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]]]:
        if flatten:
            return [mol for d in self._data for mol in d.mol]
        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self) -> List[BatchMolGraph]:
        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []

            # Dataset Loop
            for d in self._data:
                mol_graphs_list = []

                # Datapoint Loop
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        if len(d.smiles) > 1 and d.atom_features is not None:
                            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                                      'per input (i.e., number_of_molecules = 1).')
                        mol_graph = MolGraph(m, d.atom_features)                                                         # MolGraph
                        if cache_graph():
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)

                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]           # BatchMolGraph
        return self._batch_graph

    def features(self) -> Optional[List[np.ndarray]]:
        if len(self._data) == 0 or self._data[0].features is None:
            return None
        return [d.features for d in self._data]

    def atom_descriptors(self) -> Optional[List[np.ndarray]]:
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None
        return [d.atom_descriptors for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        return [d.targets for d in self._data]

    def num_tasks(self) -> int:
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        return len(self._data[0].atom_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        return len(self._data[0].atom_features[0]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def normalize_features(self,
                           scaler: StandardScaler = None,
                           replace_nan_token: int = 0) -> StandardScaler:

        if len(self._data) == 0 or self._data[0].features is None:
            return None
        if scaler is not None:
            self._scaler = scaler
        elif self._scaler is None:
            features = np.vstack([d.raw_features for d in self._data])
            self._scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self._scaler.fit(features)
        for d in self._data:
            d.set_features(self._scaler.transform(d.raw_features.reshape(1, -1))[0])
        return self._scaler

    def normalize_targets(self) -> StandardScaler:
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)
        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        return self._data[item]

