from datetime import datetime
import torch
import numpy as np
import sys
import os
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

'''
This section of code was adapted from the Chemprop project
Original Chemprop code is under the MIT License:
https://github.com/chemprop/chemprop/blob/9bc0d0ef483bd6e43ab097bbb5b93a7b065f1fa2/chemprop/utils.py
'''

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
