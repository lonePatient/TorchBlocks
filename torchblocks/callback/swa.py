import os
import copy
import torch
import logging
from torchblocks.utils.paths import create_dir, save_model
from torchblocks.utils.paths import find_all_checkpoints

logger = logging.getLogger(__name__)

#TODO 待优化
class SWA:
    '''
    checkpoint_dir:模型目录
    monitor：排序对象，按照step还是metric
    sort_mode：如果monitor=metric，则需要制定排序，最大还是最小
    k_best_checkpoints：多少个模型
    '''
    monitor_list = ['metric', 'step']
    mode_list = ['max', 'min']

    def __init__(self, checkpoint_dir,
                 monitor='step',
                 sort_mode='max',
                 device='cpu',
                 k_best_checkpoints=0,
                 checkpont_weights=None,
                 checkpoint_dir_prefix='checkpoint',
                 checkpoint_name='pytorch_model.bin'):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.sort_mode = sort_mode
        self.monitor = monitor
        self.k_best_checkpoints = k_best_checkpoints
        self.checkpoint_dir_prefix = checkpoint_dir_prefix
        self.device = torch.device(device)
        self.weights = self.init_checkpint_weight(checkpont_weights)

        if sort_mode not in self.mode_list:
            raise ValueError(f"mode: expected one of {', '.join(self.mode_list)}")

        if monitor not in self.monitor_list:
            raise ValueError(f"monitor: expected one of {', '.join(self.monitor_list)}")

    def init_checkpint_weight(self, weights):
        if not isinstance(weights, list):
            weights = [1. / (n + 1.) for n in range(self.k_best_checkpoints)]
        return weights

    def get_model_path_list(self):
        try:
            model_lists = find_all_checkpoints(checkpoint_dir=self.checkpoint_dir,
                                               checkpoint_prefix=self.checkpoint_dir_prefix,
                                               checkpoint_name=self.checkpoint_name)
            if self.monitor == 'step':
                model_lists = sorted(model_lists,
                                     key=lambda x: int(x.split("/")[-2].split("-")[-1]))
            elif self.monitor == 'metric':
                is_reverse = False
                if self.sort_mode == 'min': is_reverse = True
                model_lists = sorted(model_lists,
                                     key=lambda x: float(x.split("/")[-2].split("-")[-1][2]),
                                     reverse=is_reverse)
            model_lists = model_lists[-self.k_best_checkpoints:]
            logger.info(f"Averaging checkpoints: {[f.split('/')[-2] for f in model_lists]}")
            return model_lists
        except Exception as e:
            logger.info("Error in `swa.get_model_path_list")
            print(e)

    def step(self, model):
        """
        swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
        """
        model_path_list = self.get_model_path_list()
        swa_model = copy.deepcopy(model)
        with torch.no_grad():
            for indx, _ckpt in enumerate(model_path_list):
                logger.info(f'Load model from {_ckpt}')
                model.load_state_dict(torch.load(_ckpt, map_location=self.device))
                tmp_para_dict = dict(model.named_parameters())
                alpha = self.weights[indx]
                if indx == 0:
                    for name, para in swa_model.named_parameters():
                        para.copy_(tmp_para_dict[name].data.clone() * alpha)
                else:
                    for name, para in swa_model.named_parameters():
                        para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
        swa_model_dir = os.path.join(self.checkpoint_dir, f'checkpoint-swa-{self.sort_mode}-{self.k_best_checkpoints}')
        create_dir(swa_model_dir)
        logger.info(f'Save swa model in: {swa_model_dir}')
        swa_model_path = os.path.join(swa_model_dir, self.checkpoint_name)
        save_model(swa_model.state_dict(), swa_model_path)
        return swa_model
