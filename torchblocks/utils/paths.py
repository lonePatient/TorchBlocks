import os
import time
import glob
import torch
import json
import pickle
import logging
import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)

def get_files(directory):
    directory = os.path.expanduser(directory)
    return [os.path.join(directory, fname) for fname in os.listdir(directory)]

def check_dir(d):
    if not os.path.exists(d):
        print("Directory {} does not exist. Exit.".format(d))
        exit(1)


def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)


def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)


def get_checkpoints(output_dir, checkpoint_number, weight_name):
    '''
    获取所有checkpoint模型目录
    :param output_dir:
    :param checkpoint_number: 指定checkpoint number模型目录
    :param weight_name:
    :return:
    '''
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + weight_name, recursive=True))
    )
    checkpoints = [x for x in checkpoints if "checkpoint" in x]
    if len(checkpoints) == 0:
        raise ValueError('You need to save some checkpoints of model')
    if checkpoint_number > 0:
        checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(checkpoint_number)]
        print("Successfully loaded checkpoints.")
    return checkpoints


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    with open(str(file_path), 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def save_numpy(data, file_path):
    '''
    保存成.npy文件
    :param data:
    :param file_path:
    :return:
    '''
    np.save(str(file_path), data)


def load_numpy(file_path):
    '''
    加载.npy文件
    :param file_path:
    :return:
    '''
    np.load(str(file_path))


def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def json_to_text(file_path, data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def dict_to_text(file_path, data):
    '''
    将dict写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    with open(str(file_path), 'w') as fw:
        for key in sorted(data.keys()):
            fw.write("{} = {}\n".format(key, str(data[key])))


def save_model(model, model_path):
    """ 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_name:
    :param only_param:
    :return:
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)


def load_model(model, model_path, key='state_dict'):
    '''
    load model
    :param model:
    :param model_name:
    :param model_path:
    :param only_param:
    :return:
    '''
    logger.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states[key]
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model
