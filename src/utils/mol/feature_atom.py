from rdkit import Chem
from typing import List
from src.utils.operation import onek_encoding_unk

'''
This section of code was adapted from the Chemprop project
Original Chemprop code is under the MIT License:
https://github.com/chemprop/chemprop/blob/d2b243939f12e22b3a1d0a4b2d3599852975cf2b/chemprop/features/featurization.py
'''

MAX_ATOMIC_NUM = 100
EXTRA_ATOM_FDIM = 0

ATOM_FEATURES = {
    'atomic_num'   : list(range(MAX_ATOMIC_NUM)),
    'degree'       : [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag'   : [0, 1, 2, 3],
    'num_Hs'       : [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2

def get_atom_fdim() -> int:
    return ATOM_FDIM + EXTRA_ATOM_FDIM

def atom_features(atom: Chem.rdchem.Atom,
                  functional_groups: List[int] = None):
    features = onek_encoding_unk(atom.GetAtomicNum() - 1,      ATOM_FEATURES['atomic_num'])    + \
               onek_encoding_unk(atom.GetTotalDegree(),        ATOM_FEATURES['degree'])        + \
               onek_encoding_unk(atom.GetFormalCharge(),       ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()),     ATOM_FEATURES['chiral_tag'])    + \
               onek_encoding_unk(int(atom.GetTotalNumHs()),    ATOM_FEATURES['num_Hs'])        + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features
