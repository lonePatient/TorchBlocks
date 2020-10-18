import os
import time
import json
import random
import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def print_config(config):
    '''
    print config infomation
    '''
    config = config.__dict__
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"
    print("\n" + info + "\n")


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(_Encoder, self).default(obj)


def to_json_string(data: Dict):
    """Serializes this instance to a JSON string."""
    return json.dumps(data, indent=2, sort_keys=True, cls=_Encoder)


def _select_seed_randomly(min_seed_value=0, max_seed_value=255):
    seed = random.randint(min_seed_value, max_seed_value)
    logger.warning(f"No correct seed found, seed set to {seed}")
    return seed


def seed_everything(seed=None, deterministic_cudnn=False):
    '''
    Setting multiple seeds to make runs reproducible.

    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA,
    but might slow down your training (see https://pytorch.org/docs/stable/notes/randomness.html#cudnn) !
    :param seed:number to use as seed
    :type seed: int
    :param deterministic_torch: Enable for full reproducibility when using CUDA. Caution: might slow down training.
    :type deterministic_cudnn: bool
    :return: None
    '''
    if seed is None:
        seed = int(_select_seed_randomly())
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    '''
    computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_or_initialize_parameters(model_path, model):
    '''
    加载模型or初始化模型参数
    Args:
        model_path:
        model:
    Returns:
    '''
    if model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)


def prepare_device(use_gpu, local_rank=-1):
    """
    setup GPU device if available, move model into configured device
    # 如果n_gpu_use为数字，则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
    Example:
        use_gpu = '' : cpu
        use_gpu = '0': cuda:0
        use_gpu = '0,1' : cuda:0 and cuda:1
     """
    if local_rank == -1:
        n_gpu_use = [int(x) for x in use_gpu.split(",")]
        if len(n_gpu_use) == 0:
            device_type = 'cpu'
        else:
            device_type = f"cuda:{n_gpu_use[0]}"
        n_gpu = torch.cuda.device_count()
        if len(n_gpu_use) > 0 and n_gpu == 0:
            logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            device_type = 'cpu'
        if len(n_gpu_use) > n_gpu:
            msg = f"Warning: The number of GPU\'s configured to use is {n_gpu}, but only {n_gpu} are available on this machine."
            logger.warning(msg)
            n_gpu_use = range(n_gpu)
        device = torch.device(device_type)
        n_gpu = len(n_gpu_use)
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   local_rank, device, n_gpu, bool(local_rank != -1))
    return device, n_gpu


class Timer(object):
    """
    Record multiple running times.
        c = torch.zeros(n)
        timer = Timer()
        for i in range(n):
            c[i] = a[i] + b[i]
        f'{timer.stop():.5f} sec'
    """

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated times."""
        return np.array(self.times).cumsum().tolist()
