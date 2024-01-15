from utils.BaseFlags import parser as parser

parser.add_argument('--dataset', type=str, default='AAD', help="name of the dataset")

parser.add_argument('--style_dim', type=int, default=0, help="style dimensionality")
parser.add_argument('--num_classes', type=int, default=2, help="number of classes on which the data set trained")
parser.add_argument('--likelihood', type=str, default='normal', help="output distribution")
parser.add_argument('--gpu', type=int, default=5, help="gpu device indices")
parser.add_argument('--shell_output', type=str, default='shell_output', help="shell output save path")
parser.add_argument('--visualize_path', type=str, default='visualize', help="visualize save path")

# contras
parser.add_argument('--temperature', type=float, default=1, help="temperature of contrastive loss")

#MoPoE
parser.add_argument('--rec_weight', type=float, default=1, help="weight of reconstruction loss")
parser.add_argument('--KLD_weight', type=float, default=1, help="weight of KLD loss")
parser.add_argument('--clf_weight', type=float, default=1, help="weight of clf loss")
parser.add_argument('--contrastive_weight', type=float, default=1, help="weight of contrastive loss")
parser.add_argument('--clf_dp_prob', type=float, default=0, help="dropout prob for clf")
parser.add_argument('--aud_dp_prob', type=float, default=0, help="dropout prob for audio")
parser.add_argument('--eeg_dp_prob', type=float, default=0, help="dropout prob for clf")

# AAD
parser.add_argument('--train_clf', default=False, action="store_true",
                    help="flag to indicate if a classifier need to be train")
parser.add_argument('--clf_AAD_save', type=str, default='clf_AAD', help="model save for vae_bimodal")
parser.add_argument('--notes', type=str, default="test_baseline", help="notes to explain the experiment")
parser.add_argument('--clf_epoch', type=int, default=50, help="max epochs for training classification of AAD")
parser.add_argument('--train_clf_only', action='store_true', help="train the classification only")
parser.add_argument('--vae_model_path', type=str, default='/data/home/chenxiaoyu/code/VAE/MoPoE/runs/tmp/AAD_2022_10_15_14_08_35_016971/checkpoints/0099/mm_vae', help="model save path for MoPoE")
parser.add_argument('--sub_num', type=int, default=16, help="number of subject")

# data
parser.add_argument('--unimodal-datapaths-train', nargs="+", type=str, help="directories where training data is stored")
parser.add_argument('--unimodal-datapaths-test', nargs="+", type=str, help="directories where test data is stored")
parser.add_argument('--pretrained-classifier-paths', nargs="+", type=str, help="paths to pretrained classifiers")
parser.add_argument('--dataset_partition_file', type=str, help="paths to dataset partition file")
parser.add_argument('--win_length', type=int, default=3, help="window length for each data point")
parser.add_argument('--eeg_channel', type=int, default=10, help="channel of EEG signal")
parser.add_argument('--eeg_band', type=int, default=4, help="bands of EEG signal")
