from src.utils.mol.feature_atom import get_atom_fdim
from rdkit import Chem
from typing import List, Union
from src.utils.operation import onek_encoding_unk

'''
This section of code was adapted from the Chemprop project
Original Chemprop code is under the MIT License:
https://github.com/chemprop/chemprop/blob/d2b243939f12e22b3a1d0a4b2d3599852975cf2b/chemprop/features/featurization.py
'''

BOND_FDIM = 14

def get_bond_fdim(atom_messages: bool = False) -> int:
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond