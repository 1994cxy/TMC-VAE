import os

import random
import numpy as np
from itertools import chain, combinations

import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score

from modalities.AAD import AAD

from AADDataset import AADDataset

from MMVAE.networks.VAEAAD import VAEAAD


from MMVAE.networks.ConvNetworksEEG import EncoderEEG, DecoderEEG
from MMVAE.networks.ConvNetworksAudio import EncoderAud, DecoderAud
from MMVAE.networks.ClfAAD import ClfAAD

from utils.BaseExperiment import BaseExperiment


class AADExperiment(BaseExperiment):
    def __init__(self, flags):
        super().__init__(flags)
        self.num_modalities = flags.num_mods

        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()

        self.mm_vae = self.set_model()
        self.clf_AAD = self.set_clf_AAD()
        self.optimizer = None
        self.optimizer_clfs = None
        self.best_clf_Acc = 0.9

    def set_model(self):
        model = VAEAAD(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device)
        return model

    def set_modalities(self):
        """
        mods = [instances of modality, ]
        modality: a wrapper contains encoder and decoder of each possible modality combination.

        Returns:

        """
        mods = [AAD("m0", EncoderEEG(self.flags),
                       DecoderEEG(self.flags), self.flags.class_dim,
                       self.flags.style_dim, self.flags.likelihood),
                AAD("m1", EncoderAud(self.flags),
                       DecoderAud(self.flags), self.flags.class_dim,
                       self.flags.style_dim, self.flags.likelihood),
                AAD("m2", EncoderAud(self.flags),
                    DecoderAud(self.flags), self.flags.class_dim,
                    self.flags.style_dim, self.flags.likelihood)
                ]
        mods_dict = {m.name: m for m in mods}
        return mods_dict

    def set_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_csv = os.path.join(self.flags.dataset_partition_file, 'train.csv')
        test_csv = os.path.join(self.flags.dataset_partition_file, 'test.csv')
        train = AADDataset(self.flags.unimodal_datapaths_train, train_csv, self.flags, transform=transform)
        test = AADDataset(self.flags.unimodal_datapaths_test, test_csv, self.flags, transform=transform)
        self.dataset_train = train
        self.dataset_test = test
        print(f'training samples:{len(self.dataset_train)}')
        print(f'testing samples:{len(self.dataset_test)}')
        test_sub = []
        for sub_idx in range(1, self.flags.sub_num + 1):
            test = AADDataset(self.flags.unimodal_datapaths_test, test_csv, self.flags, transform=transform,
                              sub_id=sub_idx)
            test_sub.append(test)
        self.dataset_test_sub = test_sub

    def set_clf_AAD(self):
        """
        set a classifier for AAD
        Returns:

        """
        # clfs = {"m%d" % m: None for m in range(self.num_modalities)}
        if self.flags.train_clf:
            model_clf = ClfAAD(self.flags)
            model_clf = model_clf.to(self.flags.device)

        return model_clf

    def set_optimizer(self):
        # optimizer definition
        total_params = sum(p.numel() for p in self.mm_vae.parameters())
        params = list(self.mm_vae.parameters());
        print('num parameters: ' + str(total_params))
        optimizer = optim.Adam(params,
                               lr=self.flags.initial_learning_rate,
                               betas=(self.flags.beta_1,
                               self.flags.beta_2))
        self.optimizer = optimizer

        # optimizer for classifier of AAD
        if self.flags.train_clf:
            total_params = sum(p.numel() for p in self.clf_AAD.parameters())
            params = list(self.clf_AAD.parameters());
            print('num parameters clf: ' + str(total_params))
            optimizer_clf = optim.Adam(params,
                                   lr=self.flags.initial_learning_rate,
                                   betas=(self.flags.beta_1,
                                          self.flags.beta_2))
            self.optimizer_clf = optimizer_clf

