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
