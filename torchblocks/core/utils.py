import importlib.util
def is_apex_available():
    return importlib.util.find_spec("apex") is not None
