import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def select_seed_randomly(min_seed_value=0, max_seed_value=1024):
    seed = random.randint(min_seed_value, max_seed_value)
    logger.warning((f"No seed found, seed set to {seed}"))
    return int(seed)


def seed_everything(seed=None,verbose=True):
    '''
    init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    '''
    if seed is None: seed = select_seed_randomly()
    if verbose:
        logger.info(f"Global seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if verbose:
            logger.info("cudnn is enabled.")
