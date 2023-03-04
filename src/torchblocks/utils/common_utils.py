import datetime


def convert_to_list(obj):
    """
    Converts to list if given object is not a list.
    """
    if not isinstance(obj, list):
        obj = [obj]
    return obj


def check_object_keys(object, key, msg):
    '''
    object包含key，否则报错
    Args:
        object:
        key:
        msg:
    Returns:
    '''
    if key not in object:
        msg = (f"There were expected keys in the {msg}: "
               f"{', '.join(list(object.keys()))}, "
               f"but get {key}."
               )
        raise ValueError(msg)


def check_object_type(object, check_type, name, prefix=None):
    '''
    object满足check_type类型，否则报错
    Args:
        object:
        check_type:
        name:
        prefix:
    Returns:
    '''
    if not isinstance(object, check_type):
        msg = f"The type of {name} must be {check_type}, but got {type(object)}."
        if prefix is not None:
            msg += f' And {prefix}'
        raise TypeError(msg)


def build_datetime_str():
    """Create a string indicating current time
    Returns:
        str: current time string
    """
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime('%y%m%d%H%M%S')
    return datetime_str


def has_key(_dict, key):
    if isinstance(_dict, dict) and key in _dict:
        return True
    else:
        return False


def has_keys(_dict, *keys):
    """Check whether a nested dict has a key
    Args:
        _dict (Dict): a nested dict like object
        *keys (str): flattened key list
    Returns:
        bool: whether _dict has keys
    """
    if not _dict or not keys:
        return False
    sub_dict = _dict
    for key in keys:
        if isinstance(sub_dict, dict) and key in sub_dict:
            sub_dict = sub_dict[key]
        else:
            return False
    return True
