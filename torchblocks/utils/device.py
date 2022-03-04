import torch
import logging

logger = logging.getLogger(__name__)


def prepare_device(device_id):
    """
    setup GPU device if available, move model into configured device
    # 如果输入的是一个list，则默认使用list[0]作为controller
    Example:
        device_id = 'cpu' : cpu
        device_id = '0': cuda:0
        device_id = '0,1' : cuda:0 and cuda:1
     """
    if not isinstance(device_id, str):
        msg = 'device_id should be a str,e.g. multi-gpu:"0,1,.." or single-gpu:"0" or cpu:"cpu"'
        raise TypeError(msg)
    machine_device_num = get_all_available_gpus()
    if machine_device_num == 0 or device_id == 'cpu':
        device_num = 0
        device = torch.device('cpu')
        msg = "Warning: There\'s no GPU available on this machine, training will be performed on CPU."
        logger.warning(msg)
    else:
        logger.info(f"Available GPU\'s: {machine_device_num}")
        device_ids = [int(x) for x in device_id.split(",")]
        device_num = len(device_ids)
        device_type = f"cuda:{device_ids[0]}"
        device = torch.device(device_type)
        if device_num > machine_device_num:
            msg = (f"The number of GPU\'s configured to use is {device_num}, "
                   f"but only {machine_device_num} are available on this machine."
                   )
            logger.warning(msg)
            device_num = machine_device_num
    logger.info("Finally, device: %s, n_gpu: %s", device, device_num)
    return device, device_num


def get_all_available_gpus():
    return torch.cuda.device_count()
