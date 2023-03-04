import os
import torch
import glob
import logging
import torch.nn as nn
from .io_utils import check_file, build_dir

logger = logging.getLogger()


def save_model(model, file_path):
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    torch.save(state_dict, file_path)


def load_model(model, file_path, device=None):
    check_file(file_path)
    logger.info(f"loading model from {str(file_path)} .")
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
    build_dir(save_dir)
    torch.jit.save(traced_model, os.path.join(save_dir, 'pytorch_model.ts'))
    return save_dir


def find_all_checkpoints(ckpt_dir,
                         ckpt_prefix='checkpoint',
                         ckpt_postfix='-step-',
                         ckpt_name='pytorch_model.bin',
                         ckpt_custom_names=None):
    ckpt_list = list(
        os.path.dirname(c) for c in sorted(glob.glob(ckpt_dir + "/**/" + ckpt_name, recursive=True))
    )
    ckpt_list = [x for x in ckpt_list if ckpt_prefix in x and ckpt_postfix in x]
    if len(ckpt_list) == 0:
        raise ValueError(f"No checkpoint found at : '{ckpt_dir}'")
    if ckpt_custom_names is not None:
        if not isinstance(ckpt_custom_names, list):
            ckpt_custom_names = [ckpt_custom_names]
        ckpt_list = [x for x in ckpt_list if x.split('/')[-1] in ckpt_custom_names]
        logger.info(f"Successfully get checkpoints：\n{ckpt_list}.")
    return ckpt_list
