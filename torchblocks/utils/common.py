def check_object_type(object, check_type, name):
    if not isinstance(object, check_type):
        raise TypeError(f"The type of {name} must be {check_type}, but got {type(object)}.")
