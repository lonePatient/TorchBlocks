import os
import torch
import json
import pickle
import logging
import numpy as np
import yaml

logger = logging.getLogger()


def is_file(file_path):
    if os.path.isfile(file_path):
        return True
    return False


def is_dir(file_path):
    if os.path.isdir(file_path):
        return True
    return False


def check_file(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"File is not found here: {file_path}")
    return True


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory is not found here: {dir_path}")
    return True


def find_all_files(dir_path):
    dir_path = os.path.expanduser(dir_path)
    files = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]
    logger.info(f"The number of files: {len(files)} , Direcory:{dir_path}")
    return files


def build_dir(dir_path, exist_ok=True):
    if os.path.isdir(dir_path) and os.path.exists(dir_path):
        logger.info(f"Directory {dir_path} exist. ")
    os.makedirs(dir_path, exist_ok=exist_ok)


def save_pickle(data, file_path):
    with open(str(file_path), 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(str(file_path), 'rb') as f:
        data = pickle.load(f)
    return data


def save_numpy(data, file_path):
    np.save(str(file_path), data)


def load_numpy(file_path):
    np.load(str(file_path))


def save_json(data, file_path):
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def load_json(file_path):
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def to_json_string(data):
    """Serializes this instance to a JSON string."""
    return json.dumps(data, indent=2, sort_keys=True, cls=_Encoder)


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(_Encoder, self).default(obj)


def json_to_text(file_path, data):
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def dict_to_text(file_path, data):
    with open(str(file_path), 'w') as fw:
        for key in sorted(data.keys()):
            fw.write("{} = {}\n".format(key, str(data[key])))


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.load(f)
    return data


def save_yaml(data, file_path):
    with open(file_path, 'w') as fw:
        yaml.dump(data, fw)
