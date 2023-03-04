import os
import sys
import importlib.util
from pathlib import Path
from importlib import import_module

def is_apex_available():
    return importlib.util.find_spec("apex") is not None

def import_modules_from_file(py_file: str):
    """ Import module from a certrain file
    Args:
        py_file: path to a python file to be imported
    Return:
    """
    dirname, basefile = os.path.split(py_file)
    if dirname == '':
        dirname = Path.cwd()
    module_name = os.path.splitext(basefile)[0]
    sys.path.insert(0, dirname)
    mod = import_module(module_name)
    sys.path.pop(0)
    return module_name, mod
