
from abc import ABC, abstractmethod

import os
from itertools import chain, combinations

class BaseExperiment(ABC):
    def __init__(self, flags):
        self.flags = flags;
        self.name = flags.dataset;

        self.modalities = None;
        self.num_modalities = None;
        self.subsets = None;
        self.dataset_train = None;
        self.dataset_test = None;

        self.mm_vae = None;
        self.clfs = None;
        self.optimizer = None;


    @abstractmethod
    def set_model(self):
        pass;

    @abstractmethod
    def set_modalities(self):
        pass;

    @abstractmethod
    def set_dataset(self):
        pass;

    # @abstractmethod
    # def set_clfs(self):
    #     pass;

    @abstractmethod
    def set_optimizer(self):
        pass;

    def set_subsets(self):
        num_mods = len(list(self.modalities.keys()));

        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
        xs = list(self.modalities)
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                          range(len(xs)+1))
        subsets = dict();
        for k, mod_names in enumerate(subsets_list):
            mods = [];
            for l, mod_name in enumerate(sorted(mod_names)):
                mods.append(self.modalities[mod_name])
            key = '_'.join(sorted(mod_names));
            subsets[key] = mods;
        return subsets;
