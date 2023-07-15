from rdkit import Chem
from rdkit.Chem import PandasTools
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import csv
import sys
import os
import pickle
import pandas as pd
from typing import List
from argparse import Namespace


def Print(string,
          output=sys.stdout,
          newline=False):
    """ print to stdout and a file (if given) """
    time = datetime.now()
    print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=sys.stderr)
    if newline:
        print("", file=sys.stderr)

    if not output == sys.stdout:
        print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=output)
        if newline:
            print("", file=output)

    output.flush()
    return time


def read_fasta(file):
    fasta = ""
    with open(file, "r") as f:
        for line in f:
            if line[0] == ">":
                continue
            else:
                line = line.rstrip()
                fasta = fasta + line
    return fasta


def set_output(args,
               string,
               test=False):
    """ set output configurations """
    output, save_prefix, index = sys.stdout, None, ""
    if args["output_path"] is not None:
        if not test:
            if not os.path.exists(args["output_path"] + "/weights/"):
                os.makedirs(args["output_path"] + "/weights/", exist_ok=True)
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/weights/" + index

        else:
            if not os.path.exists(args["output_path"]):
                os.makedirs(args["output_path"], exist_ok=True)
            if args["pretrained_model"] is not None:
                index += os.path.splitext(args["pretrained_model"])[0].split("/")[-1] + "_"
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")

    return output, save_prefix


def load_pretrained_models(args,
                           models,
                           device,
                           data_parallel,
                           output,
                           tfm_cls=True):
    """ load models if pretrained_models are available """
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_model" if idx == "" else "pretrained_%s_model" % idx
        if idx in args and args[idx] is not None:
            Print('loading %s weights from %s' % (idx, args[idx]), output)
            if not tfm_cls and idx == "":
                models[m][0].load_pretrained_weights(args[idx], tfm_cls)
            else:
                models[m][0].load_pretrained_weights(args[idx])

        models[m][0] = models[m][0].to(device)
        if data_parallel:
            models[m][0] = nn.DataParallel(models[m][0])


def load_features(path: str) -> np.ndarray:
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['feature']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def load_valid_atom_features(path: str, smiles: List[str]) -> List[np.ndarray]:
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        container = np.load(path)
        features = [container[key] for key in container]

    elif extension in ['.pkl', '.pckl', '.pickle']:
        features_df = pd.read_pickle(path)
        if features_df.iloc[0, 0].ndim == 1:
            features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()
        elif features_df.iloc[0, 0].ndim == 2:
            features = features_df.apply(lambda x: np.concatenate(x.tolist(), axis=1), axis=1).tolist()
        else:
            raise ValueError(f'Atom descriptors input {path} format not supported')

    elif extension == '.sdf':
        features_df = PandasTools.LoadSDF(path).drop(['ID', 'ROMol'], axis=1).set_index('SMILES')

        features_df = features_df[~features_df.index.duplicated()]

        # locate atomic descriptors columns
        features_df = features_df.iloc[:,
                      features_df.iloc[0, :].apply(lambda x: isinstance(x, str) and ',' in x).to_list()]
        features_df = features_df.reindex(smiles)
        if features_df.isnull().any().any():
            raise ValueError('Invalid custom atomic descriptors file, Nan found in data')

        features_df = features_df.applymap(
            lambda x: np.array(x.replace('\r', '').replace('\n', '').split(',')).astype(float))

        # Truncate by number of atoms
        num_atoms = {x: Chem.MolFromSmiles(x).GetNumAtoms() for x in features_df.index.to_list()}

        def truncate_arrays(r):
            return r.apply(lambda x: x[:num_atoms[r.name]])

        features_df = features_df.apply(lambda x: truncate_arrays(x), axis=1)

        features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()

    else:
        raise ValueError(f'Extension "{extension}" is not supported.')

    return features


def makedirs(path: str,
             isfile: bool = False) -> None:
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def save_features(path: str, features: List[np.ndarray]) -> None:
    np.savez_compressed(path, features=features)


def save_checkpoint(path: str,
                    model: torch.nn.Module,
                    model_type: str,
                    args):
    if args is not None:
        args = Namespace(**args.as_dict())

    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'model_type': model_type,
    }
    torch.save(state, path)


def load_checkpoint(path,
                    logger=None,
                    save_dir=None,
                    to_cpu=False):
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug, info = Print, Print

    state = torch.load(path, map_location=lambda storage, loc: storage)
    model_type = state["model_type"]
    args = select_args(model_type)()
    args.from_dict(vars(state['args']), skip_unsettable=True)

    if to_cpu is True:
        args.no_cuda = True

    loaded_state_dict = state['state_dict']

    model = select_model(model_type)(args, save_dir=save_dir)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            info(f'Warning: Pretrained parameter "{param_name}" '
                 f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def select_model(model_type):
    if model_type == 'attentivefp':
        from src.model.attentivefp import attentivefp
        return attentivefp
    elif model_type == "attentivefp_postnet":
        from src.model.attentivefp_postnet import attentivefpPostNet
        return attentivefpPostNet

def select_args(model_type):
    if model_type == 'attentivefp':
        from src.config.attentivefp import attentivefpArgs
        return attentivefpArgs
    elif model_type == 'attentivefp_postnet':
        from src.config.attentivefp_postnet import attentivefpPostNetArgs
        return attentivefpPostNetArgs
