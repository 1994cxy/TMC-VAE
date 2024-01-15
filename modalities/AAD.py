
import torch

from modalities.Modality import Modality


class AAD(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        """
        encoder and decoder for one modality
        Args:
            name: modality name
            enc: encoder instance
            dec: decoder instance
            class_dim: #todo
            style_dim: #todo
        """
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name)
