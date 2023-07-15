from .io import makedirs, Print
import os
import json
import numpy as np

class Writer:
    def __init__(self, file_path):
        self.file_path = file_path
        makedirs(os.path.dirname(file_path))

    def __call__(self, message):
        with open(self.file_path, "a+") as f:
            f.write(message+"\n")
            Print(message)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)