

import argparse

parser = argparse.ArgumentParser()

# DATASET NAME
# to be specified by experiments themselves
# parser.add_argument('--dataset', type=str, default='SVHN_MNIST_text', help="name of the dataset")

# TRAINING
parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")


#multimodal
parser.add_argument('--method', type=str, default='poe', help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False, help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False, help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False, help="modality_moe")
parser.add_argument('--joint_elbo', type=bool, default=False, help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=True, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False, help="factorized_representation")


# DATA DEPENDENT
parser.add_argument('--class_dim', type=int, default=128, help="dimension of common factor latent space")

# SAVE and LOAD
parser.add_argument('--mm_vae_save_name', type=str, default='AE', help="model save for vae")
parser.add_argument('--clf_save_name', type=str, default='AE_clf', help="clf model save filename")
parser.add_argument('--load_saved', action='store_true', default=False, help="flag to indicate if a saved model will be loaded")

# DIRECTORIES
#exp
parser.add_argument('--dir_experiment', type=str, default='/tmp/multilevel_multimodal_vae_swapping', help="directory to save generated samples in")

# data
parser.add_argument('--dir_data', type=str, default='../data', help="directory where data is stored")

# EVALUATION
parser.add_argument('--eval_freq', type=int, default=10, help="frequency of evaluation of latent representation of generative performance (in number of epochs)")


# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=5.0, help="default weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=1.0, help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0, help="default weight of sum of weighted content divergence terms")

