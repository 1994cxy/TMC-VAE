
import sys
import os
import json
import torch
import warnings

from run_epochs_AAD import run_epochs
from utils.utils import save_paramters
from utils.filehandling import create_dir_structure
from MMVAE.MMVAE_flags import parser
from MMVAE.experiment_MMVAE import AADExperiment

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)

    FLAGS = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device(f'cuda:{FLAGS.gpu}' if use_cuda else 'cpu')

    # print(FLAGS.unimodal_datapaths_train)
    # print(FLAGS.unimodal_datapaths_test)
    # print(f'class_dim:{FLAGS.class_dim}')
    # print(f'initial_learning_rate:{FLAGS.initial_learning_rate}')
    # print(f'initial_learning_rate:{FLAGS.initial_learning_rate}')
    # print(f'end_epoch:{FLAGS.end_epoch}')

    # postprocess flags
    assert len(FLAGS.unimodal_datapaths_train) == len(FLAGS.unimodal_datapaths_test)
    FLAGS.num_mods = len(FLAGS.unimodal_datapaths_train)  # set number of modalities dynamically


    FLAGS.alpha_modalities = [FLAGS.num_mods + 1]
    FLAGS.alpha_modalities.extend([FLAGS.num_mods + 1 for _ in range(FLAGS.num_mods)])
    # print("alpha_modalities:", FLAGS.alpha_modalities)
    create_dir_structure(FLAGS)

    aad = AADExperiment(FLAGS)
    aad.set_optimizer()

    save_paramters(FLAGS)

    run_epochs(aad)
