import os
import torch
import json
import pickle
import logging
import glob
import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)


def check_file(file_path):
    if not os.path.isfile(file_path):
        raise ValueError(f"File is not found here: {file_path}")
    return True


def is_file(file_path):
    if os.path.isfile(file_path):
        return True
    return False


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory is not found here: {dir_path}")
    return True


def find_all_files(dir_path):
    dir_path = os.path.expanduser(dir_path)
    files = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]
    logger.info(f"The number of files: {len(files)} , Direcory:{dir_path}")
    return


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Directory {dir_path} do not exist; creating...")


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


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(_Encoder, self).default(obj)


def to_json_string(data):
    """Serializes this instance to a JSON string."""
    return json.dumps(data, indent=2, sort_keys=True, cls=_Encoder)


def json_to_text(file_path, data):
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def dict_to_text(file_path, data):
    with open(str(file_path), 'w') as fw:
        for key in sorted(data.keys()):
            fw.write("{} = {}\n".format(key, str(data[key])))


def save_model(model, file_path):
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    torch.save(state_dict, file_path)


def load_model(model, file_path, device=None):
    if check_file(file_path):
        print(f"loading model from {str(file_path)} .")
    state_dict = torch.load(file_path, map_location="cpu" if device is None else device)
    if isinstance(model, nn.DataParallel) or hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)


def save_jit_model(model, example_inputs, save_dir, dir_name=None):
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs=example_inputs, strict=False)
    if dir_name is None:
        save_dir = os.path.join(save_dir, 'save_model_jit_traced')
    else:
        save_dir = os.path.join(save_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.jit.save(traced_model, os.path.join(save_dir, 'pytorch_model.ts'))
    return save_dir


def find_all_checkpoints(checkpoint_dir,
                         checkpoint_prefix='checkpoint',
                         checkpoint_name='pytorch_model.bin',
                         checkpoint_custom_names=None):
    '''
    获取模型保存路径下所有checkpoint模型路径，其中
    checkpoint_custom_names：表示自定义checkpoint列表
    '''
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(checkpoint_dir + "/**/" + checkpoint_name, recursive=True))
    )
    checkpoints = [x for x in checkpoints if checkpoint_prefix in x]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found at : '{}'".format(checkpoint_dir))
    if checkpoint_custom_names is not None:
        if not isinstance(checkpoint_custom_names, list):
            checkpoint_custom_names = [checkpoint_custom_names]
        checkpoints = [x for x in checkpoints if x.split('/')[-1] in checkpoint_custom_names]
        logger.info(f"Successfully get checkpoints：{checkpoints}.")
    return checkpoints
