"""
Utilities adapted from the TKGE project.
"""

import os
import random
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import constant_, xavier_normal_
import numpy as np
from prettytable import PrettyTable
import loralib



def set_seeds(seed):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_param(shape):
    """Get a learnable parameter tensor."""
    param = Parameter(torch.Tensor(*shape)).double()
    xavier_normal_(param.data)
    return param


