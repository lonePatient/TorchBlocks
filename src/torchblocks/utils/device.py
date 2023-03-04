import torch
import logging
from .common_utils import check_object_type

logger = logging.getLogger()


def get_all_available_gpus():
    return torch.cuda.device_count()


def build_device(device_id):
    """build torch device
    setup GPU device if available, move model into configured device
    # 如果输入的是一个list，则默认使用list[0]作为controller
    Example:
        device_id = 'cpu' : cpu
        device_id = '0': cuda:0
        device_id = '0,1' : cuda:0 and cuda:1
     """
    if isinstance(device_id, int): device_id = str(device_id)
    if '.' in device_id: device_id = device_id.repalce('.', ',')
    check_object_type(device_id, check_type=str, name='device_id')
    machine_device_num = get_all_available_gpus()
    device_type = 'cuda'
    if (
            len(device_id) == 0
            or machine_device_num == 0
            or device_id == 'cpu'
            or len(device_id.strip()) == 0
    ):
        device_type = 'cpu'
    if device_type == 'cpu':
        device_num = 0
        msg = "Warning: There\'s no GPU available on this machine, training will be performed on CPU."
        logger.warning(msg)
    else:
        logger.info(f"Available GPU\'s: {machine_device_num}")
        device_ids = [int(x) for x in device_id.split(",")]
        device_num = len(device_ids)
        device_type = f"cuda:{device_ids[0]}"
        if device_num > machine_device_num:
            msg = (f"The number of GPU\'s configured to use is {device_num}, "
                   f"but only {machine_device_num} are available on this machine."
                   )
            logger.warning(msg)
            device_num = machine_device_num
    device = torch.device(device_type)
    logger.info("Finally, device: %s, n_gpu: %s", device, device_num)
    return device, device_num


if __name__ == "__main__":
    device_id = ''
    device_id0 = ' '
    device_id1 = '0'
    device_id2 = '0,1'
    device_id3 = 'cpu'
    device_id4 = '0,1,2,3'
    device_id5 = 0
    print(build_device(device_id))
    print(build_device(device_id0))
    print(build_device(device_id1))
    print(build_device(device_id2))
    print(build_device(device_id3))
    print(build_device(device_id4))
    print(build_device(device_id5))
