from typing import Callable, List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from mordred import Calculator, descriptors

Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]

FEATURES_GENERATOR_REGISTRY = {}

def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator
    return decorator

def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')
    return FEATURES_GENERATOR_REGISTRY[features_generator_name]

def get_available_features_generators() -> List[str]:
    return list(FEATURES_GENERATOR_REGISTRY.keys())

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 1024

@register_features_generator('mordred')  # length 1215
def mordred_features_generator(mol: Molecule) -> np.ndarray:
    _calc = Calculator([descriptors.MoeType,
                        descriptors.Autocorrelation,
                        descriptors.Chi,
                        descriptors.EState,
                        descriptors.InformationContent,
                        descriptors.AdjacencyMatrix,
                        descriptors.BaryszMatrix,
                        descriptors.DetourMatrix,
                        descriptors.DistanceMatrix])
    _MatrixNames = [str(i) for i in _calc.descriptors]
    r = _calc(mol)
    r = r.fill_missing(0)
    return np.array(list(r.values()), dtype=np.float)

@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features

@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features

@register_features_generator('MACCS')
def MACCS_feature_generator(mol: Molecule) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    maccs_fps = MACCSkeys.GenMACCSKeys(mol)
    fp_arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(maccs_fps, fp_arr)
    maccs_fp = np.array(maccs_fps, dtype=np.int)
    return maccs_fp[1:]

try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]
        return features

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        return features

except ImportError:
    pass


"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""
@register_features_generator('descriptor')
def molecular_descriptor_features_generator(mol: Molecule) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    descriptor_calculator = RDKitDescriptors()
    return descriptor_calculator.featurize(mol)


class molecules2features:
    def __init__(self, features_generator_list):
        if type(features_generator_list) is str:
            features_generator_list = [features_generator_list]
        self.features_generator_list = features_generator_list

    def featurize(self, mol_list):
        features = []
        for mol in mol_list:
            feature = []
            if type(mol) == str:
                mol = Chem.MolFromSmiles(mol)
            assert mol is not None
            for f_g_n in self.features_generator_list:
                f_g = get_features_generator(f_g_n)
                feature.append(f_g(mol))
            feature = np.concatenate(feature)
            features.append(feature)
        features = np.stack(features)
        return features


class RDKitDescriptors():
  """
  RDKit descriptors.
  See http://rdkit.org/docs/GettingStartedInPython.html
  #list-of-available-descriptors.
  """
  name = 'descriptors'

  allowedDescriptors = {'MaxAbsPartialCharge', 'MinPartialCharge', 'MinAbsPartialCharge', 'HeavyAtomMolWt',
                        'MaxAbsEStateIndex', 'NumRadicalElectrons', 'NumValenceElectrons', 'MinAbsEStateIndex',
                        'MaxEStateIndex', 'MaxPartialCharge', 'MinEStateIndex', 'ExactMolWt', 'MolWt', 'BalabanJ',
                        'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
                        'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
                        'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
                        'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
                        'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
                        'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
                        'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
                        'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                        'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
                        'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
                        'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3',
                        'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                        'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
                        'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR'}

  def __init__(self):
    self.descriptors = []
    self.descList = []
    from rdkit.Chem import Descriptors
    for descriptor, function in Descriptors.descList:
      if descriptor in self.allowedDescriptors:
        self.descriptors.append(descriptor)
        self.descList.append((descriptor, function))

  def featurize(self, mol):
    """
    Calculate RDKit descriptors.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    rval = []
    for desc_name, function in self.descList:
      rval.append(function(mol))

    return np.array(rval, np.float)

