import pandas as pd
import json
import os
import numpy as np
import csv
import math
import random
import gzip
from typing import Optional

def read_jsonl(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = []
        for line in f.readlines():
            json_data = json.loads(line)
            data.append(json_data)
    return data

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def to_jsonl(dicts, save_file):
    if not os.path.isdir(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with open(save_file, 'w', encoding='utf-8') as f:
        for line_dict in dicts:
            jsonl_line = f'{json.dumps(line_dict, cls = NpEncoder, ensure_ascii=False)}\n'
            f.write(jsonl_line)