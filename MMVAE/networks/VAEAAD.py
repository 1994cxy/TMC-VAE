import os

import torch
import torch.nn as nn

from utils import utils
from utils.BaseMMVae import BaseMMVae


class VAEAAD(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets);