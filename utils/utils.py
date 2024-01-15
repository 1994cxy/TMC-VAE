
import json
import math
import numpy as np
import torch
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def cal_conv_output_size(width, filter_size, stride, padding, dilation=1):
    filter_size = filter_size + (filter_size - 1)*(dilation - 1)
    return (width - filter_size + 2*padding)/stride + 1

def cal_deconv_output_size(width, filter_size, stride, padding, dilation=1, output_padding=0):
    return (width-1) * stride - 2*padding + dilation*(filter_size-1) + output_padding + 1

def cal_deconv_para(output_size, input_size):
    stride = math.floor(output_size / (input_size+1))
    filter_size = output_size - (input_size-1)*stride
    return filter_size, stride

def t_sne_visualize_target(gourp_a, exp, epoch=0, prefix='test', label=None):
    X = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(gourp_a)
    tsne_a = X

    fig = plt.figure()
    if label is not None:

        positive_a = tsne_a[label == 1, :]
        negative_a = tsne_a[label == 0, :]

        plt.scatter(positive_a[:, 0], positive_a[:, 1], c='b', marker='o')
        plt.scatter(negative_a[:, 0], negative_a[:, 1], c='b', marker='x')

    else:
        plt.scatter(tsne_a[:, 0], tsne_a[:, 1], c='b', marker='o')

    fig_folder = os.path.join(exp.flags.dir_experiment_run, exp.flags.visualize_path)
    plt.savefig(os.path.join(fig_folder, f'latent_scatter_{prefix}_epoch{str(epoch).zfill(4)}'))
    return fig

def t_sne_visualize(gourp_a, group_b, exp, epoch=0, prefix='test', label=None):
    origin_X = np.concatenate((gourp_a, group_b), axis=0)
    X = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(origin_X)
    tsne_a = X[:gourp_a.shape[0], :]
    tsne_b = X[gourp_a.shape[0]:, :]

    fig = plt.figure()
    if label is not None:

        positive_a = tsne_a[label==1, :]
        negative_a = tsne_a[label==0, :]
        positive_b = tsne_b[label==1, :]
        negative_b = tsne_b[label==0, :]

        plt.scatter(positive_a[:, 0], positive_a[:, 1], c='b', marker='o')
        plt.scatter(negative_a[:, 0], negative_a[:, 1], c='b', marker='x')
        plt.scatter(positive_b[:, 0], positive_b[:, 1], c='r', marker='o')
        plt.scatter(negative_b[:, 0], negative_b[:, 1], c='r', marker='x')

    else:
        plt.scatter(tsne_a[:, 0], tsne_a[:, 1], c='b', marker='o')
        plt.scatter(tsne_b[:, 0], tsne_b[:, 1], c='r', marker='x')

    fig_folder = os.path.join(exp.flags.dir_experiment_run, exp.flags.visualize_path)
    plt.savefig(os.path.join(fig_folder, f'latent_scatter_{prefix}_epoch{str(epoch).zfill(4)}'))
    return fig

def save_paramters(flags):
    """
    :param flags:
    :return:
    """
    params = dict()

    # data info
    params["unimodal-datapaths-train"] = flags.unimodal_datapaths_train
    params["unimodal-datapaths-test"] = flags.unimodal_datapaths_test
    params["dataset_partition_file"] = flags.dataset_partition_file

    # loss weight
    params["KLD_weight"] = flags.KLD_weight
    params["rec_weight"] = flags.rec_weight
    params["clf_weight"] = flags.clf_weight
    params["contrastive_weight"] = flags.contrastive_weight

    # latent dim
    params["class_dim"] = flags.class_dim

    # other params
    params["batch_size"] = flags.batch_size
    params["initial_learning_rate"] = flags.initial_learning_rate

    json_path = os.path.join(flags.dir_experiment_run, 'params.json')
    with open(json_path, 'w') as json_file:
        json_file.write(json.dumps(params))



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def save_and_log_flags(flags):
    #filename_flags = os.path.join(flags.dir_experiment_run, 'flags.json')
    #with open(filename_flags, 'w') as f:
    #    json.dump(flags.__dict__, f, indent=2, sort_keys=True)

    filename_flags_rar = os.path.join(flags.dir_experiment_run, 'flags.rar')
    torch.save(flags, filename_flags_rar);
    str_args = '';
    for k, key in enumerate(sorted(flags.__dict__.keys())):
        str_args = str_args + '\n' + key + ': ' + str(flags.__dict__[key]);
    return str_args;

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)

def reweight_weights(w):
    w = w / w.sum();
    return w;

def mixture_component_selection(flags, mus, logvars, w_modalities=None):
    #if not defined, take pre-defined weights
    num_components = mus.shape[0];
    num_samples = mus.shape[1];
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device);
    # todo: ??? why???
    idx_start = [];
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0;
        else:
            i_start = int(idx_end[k-1]);
        if k == w_modalities.shape[0]-1:
            i_end = num_samples;
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]));
        idx_start.append(i_start);
        idx_end.append(i_end);
    idx_end[-1] = num_samples;
    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    return [mu_sel, logvar_sel];