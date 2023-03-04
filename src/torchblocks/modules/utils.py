import copy
from torch.nn import ModuleList


def get_clones(module, N):
    "Produce N identical modules."
    return ModuleList([copy.deepcopy(module) for _ in range(N)])
