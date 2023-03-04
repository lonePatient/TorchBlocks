import os
import copy
import torch
from ..utils.ckpt_utils import find_all_checkpoints

def SWA(model, model_dir, swa_start=1):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = find_all_checkpoints(model_dir)
    assert 1 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'
    swa_model = copy.deepcopy(model)
    swa_n = 0.
    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            print(_ckpt)
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())
            alpha = 1. / (swa_n + 1.)
            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
            swa_n += 1
    swa_model_dir = os.path.join(model_dir, f'checkpoint-swa')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)
    swa_model_path = os.path.join(swa_model_dir, 'pytorch_model.bin')
    torch.save(swa_model.state_dict(), swa_model_path)
    return swa_model
