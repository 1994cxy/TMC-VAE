import sys, os
import numpy as np
from itertools import cycle
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import copy

from utils import utils
from utils.TBLogger_AAD import TBLogger


# global variables
SEED = None 
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

clf_loss = nn.BCELoss()
mm_vae_rec_loss = nn.MSELoss()

def pearson(x, y):
    x_reducemean = x - torch.mean(x, dim=1, keepdim=True)
    y_reducemean = y - torch.mean(y, dim=1, keepdim=True)
    numerator = torch.matmul(x_reducemean, y_reducemean.T)
    norm_x = torch.norm(x_reducemean, p=2, dim=1).unsqueeze(1)
    norm_y = torch.norm(y_reducemean, p=2, dim=1).unsqueeze(1)
    denominator = torch.matmul(norm_x, norm_y.T)
    corrcoef = numerator / denominator

    return corrcoef.diag().unsqueeze(1)


def calc_rec_audio_corr(rec_data, ori_data):
    # rec_data['m1'] = [batch, time_len, stft_point]
    # rec_data['m1'] = rec_data['m1'].mean(dim=-1)
    batch_size = ori_data['m1'].shape[0]
    rec_audio0 = rec_data['m1'].mean(dim=-1).squeeze(1)
    rec_audio1 = rec_data['m2'].mean(dim=-1).squeeze(1)
    ori_audio0 = ori_data['m1'].mean(dim=-1).squeeze(1)
    ori_audio1 = ori_data['m2'].mean(dim=-1).squeeze(1)


    # rec_audio0 = rec_data['m1'].view(batch_size, -1)
    # rec_audio1 = rec_data['m2'].view(batch_size, -1)
    # ori_audio0 = ori_data['m1'].view(batch_size, -1)
    # ori_audio1 = ori_data['m2'].view(batch_size, -1)

    person_audio0 = pearson(rec_audio0, ori_audio0)
    person_audio1 = pearson(rec_audio1, ori_audio1)

    return torch.cat((person_audio0, person_audio1), dim=1)



def calc_AAD_clf_Acc(result, label):
    total_data = len(result)
    correct = torch.sum(torch.max(result, dim=1).indices==label)
    return correct, total_data


def calc_AAD_clf_loss(result, label):
    temp_label = torch.zeros_like(result)
    for idx, item in enumerate(label):
        temp_label[idx] = torch.from_numpy(np.eye(2)[item])
    return clf_loss(result, temp_label)


def calc_mm_vae_rec_loss(rec, input):
    for m_key in rec.keys():
        i_m = rec[m_key];
        if m_key == 'm0':  # for EEG
            rec_loss = mm_vae_rec_loss(i_m, input[m_key])
        else:
            rec_loss += mm_vae_rec_loss(i_m, input[m_key])
    return rec_loss/len(rec)

def cal_contrastive_loss(z_compelete, z_target, temperature=1):
    batch_size = len(z_compelete)

    # Negative pairs: everything that is not in the current joint-modality pair
    out_joint_mod = torch.cat(
        [z_target, z_compelete], dim=0
    )
    # [2*B, 2*B]
    inner_product = torch.mm(out_joint_mod, out_joint_mod.t().contiguous())

    norm_mask = (torch.zeros_like(inner_product)
            + torch.eye(2 * batch_size, device=inner_product.device)
    ).bool()
    norm = inner_product.masked_select(
        norm_mask
    ).view(2 * batch_size, -1).repeat(1, 2 * batch_size)

    inner_product = inner_product/norm

    sim_matrix_joint_mod = torch.exp(
        inner_product / temperature
    )
    # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
    mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
    ).bool()
    # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
    sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
        mask_joint_mod
    ).view(2 * batch_size, -1)

    # Positive pairs: cosine loss joint-modality
    inner_product = torch.sum(z_target * z_compelete, dim=-1)
    norm = torch.sum(z_target * z_target, dim=-1)
    inner_product = inner_product/norm

    pos_sim_joint_mod = torch.exp(
        inner_product / temperature
    ).sum()

    loss_joint_mod = -torch.log(
        pos_sim_joint_mod / sim_matrix_joint_mod.sum()
    )

    return loss_joint_mod

def mix_audio_epoch(exp, batch):
    mm_vae = exp.mm_vae;
    batch_d = batch[0];
    batch_l = batch[1];
    audio_ori = batch[2];
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device);
    batch_l = batch_l.to(exp.flags.device);
    batch_d_mix = copy.deepcopy(batch_d)

    batch_d_mix['m1'] = (batch_d_mix['m1']+batch_d_mix['m2'])/2
    batch_d_mix.pop('m2')

    results = mm_vae.forward_mod_miss(batch_d_mix, batch_l)

    z_for_clf = results['class_embeddings']
    clf_result = exp.clf_AAD(z_for_clf)
    ADD_clf_correct, _ = calc_AAD_clf_Acc(clf_result, batch_l)

    # for joint training
    out_basic_routine = dict()
    out_basic_routine['ADD_clf_correct'] = ADD_clf_correct;
    out_basic_routine['AAD_clf_total'] = len(batch_l);
    out_basic_routine['AAD_clf_z'] = z_for_clf;
    out_basic_routine['label'] = batch_l
    return out_basic_routine;

def modality_missing_epoch(exp, batch):
    mm_vae = exp.mm_vae;
    batch_d = batch[0];
    batch_l = batch[1];
    audio_ori = batch[2];
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device);
    batch_l = batch_l.to(exp.flags.device);
    audio_ori = audio_ori.to(exp.flags.device)

    batch_d_m0 = copy.deepcopy(batch_d)
    batch_d_m0.pop('m0')
    batch_d_m1 = copy.deepcopy(batch_d)
    batch_d_m1.pop('m1')
    batch_d_m2 = copy.deepcopy(batch_d)
    batch_d_m2.pop('m2')
    batch_d_m0_only = copy.deepcopy(batch_d)
    batch_d_m0_only.pop('m1')
    batch_d_m0_only.pop('m2')

    result_dict = {'m0': mm_vae.forward_mod_miss(batch_d_m0, batch_l),
                   'm1': mm_vae.forward_mod_miss(batch_d_m1, batch_l),
                  'm2': mm_vae.forward_mod_miss(batch_d_m2, batch_l),
                  'm0_only': mm_vae.forward_mod_miss(batch_d_m0_only, batch_l)}

    ADD_clf_result = {}
    AAD_clf_z = {}
    for key, results in result_dict.items():
        z_for_clf = results['class_embeddings']
        clf_result = exp.clf_AAD(z_for_clf)
        ADD_clf_correct, _ = calc_AAD_clf_Acc(clf_result, batch_l)
        ADD_clf_result[key] = ADD_clf_correct
        AAD_clf_z[key] = z_for_clf

    # for joint training
    out_basic_routine = dict()
    out_basic_routine['ADD_clf_correct'] = ADD_clf_result;
    out_basic_routine['AAD_clf_total'] = len(batch_l);
    out_basic_routine['AAD_clf_z'] = AAD_clf_z;
    out_basic_routine['label'] = batch_l
    return out_basic_routine;

def basic_routine_epoch(exp, batch):
    mm_vae = exp.mm_vae;
    batch_d = batch[0];
    batch_l = batch[1];
    audio_ori = batch[2];
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device);
    batch_l = batch_l.to(exp.flags.device);
    audio_ori = audio_ori.to(exp.flags.device)
    results = mm_vae(batch_d, batch_l);

    # rec loss
    loss_rec = calc_mm_vae_rec_loss(results['rec_data'], batch_d)

    # KLD
    loss_kld = results['joint_divergence']

    # contrastive loss
    z_compelete = results['latent_complete']
    z_target = results['latent_target']
    loss_contrastive = cal_contrastive_loss(z_compelete, z_target, exp.flags.temperature)

    # clf loss
    z_for_clf = results['class_embeddings']
    clf_result = exp.clf_AAD(z_for_clf)
    AAD_clf_loss = calc_AAD_clf_loss(clf_result, batch_l)
    ADD_clf_correct, AAD_clf_total = calc_AAD_clf_Acc(clf_result, batch_l)

    rec_audio_corr = calc_rec_audio_corr(results['rec_data'], batch_d)
    rec_att_audio_corr = rec_audio_corr[np.arange(len(batch_l)), batch_l.cpu().numpy()].mean()
    ADD_clf_correct_use_corr, _ = calc_AAD_clf_Acc(rec_audio_corr, batch_l)

    total_loss = exp.flags.rec_weight * loss_rec + exp.flags.KLD_weight * loss_kld + exp.flags.clf_weight * AAD_clf_loss\
                 + exp.flags.contrastive_weight * loss_contrastive

    # cosine sim
    cossim = torch.cosine_similarity(results['latent_complete'], results['latent_target'])

    # for joint training
    out_basic_routine = dict()
    out_basic_routine['result'] = results
    out_basic_routine['loss_rec'] = loss_rec
    out_basic_routine['total_loss'] = total_loss
    out_basic_routine['AAD_clf_loss'] = AAD_clf_loss;
    out_basic_routine['ADD_clf_correct'] = ADD_clf_correct;
    out_basic_routine['AAD_clf_total'] = AAD_clf_total;
    out_basic_routine['loss_contrastive'] = loss_contrastive;
    out_basic_routine['mean_Pearson_corr'] = 0
    out_basic_routine['label'] = batch_l
    out_basic_routine['cossim'] = cossim
    out_basic_routine['rec_audio_corr'] = rec_audio_corr
    out_basic_routine['rec_att_audio_corr'] = rec_att_audio_corr
    out_basic_routine['ADD_clf_correct_use_corr'] = ADD_clf_correct_use_corr

    return out_basic_routine;


def train(epoch, exp, tb_logger):
    mm_vae = exp.mm_vae;
    mm_vae.train();
    exp.mm_vae = mm_vae;

    # for joint training
    clf = exp.clf_AAD;
    clf.train();
    exp.clf_AAD = clf;

    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    ADD_clf_correct, ADD_clf_correct_use_corr, AAD_clf_total = 0, 0, 0;
    for iteration, batch in enumerate(d_loader):
        basic_routine = basic_routine_epoch(exp, batch);
        total_loss = basic_routine['total_loss'];

        ADD_clf_correct += basic_routine['ADD_clf_correct']
        ADD_clf_correct_use_corr += basic_routine['ADD_clf_correct_use_corr']
        AAD_clf_total += basic_routine['AAD_clf_total']

        # backprop
        exp.optimizer.zero_grad()
        exp.optimizer_clf.zero_grad()
        total_loss.backward()
        exp.optimizer_clf.step()
        exp.optimizer.step()
        tb_logger.write_training_logs(basic_routine);

    # for joint training
    AAD_clf_Acc = ADD_clf_correct / AAD_clf_total
    AAD_clf_Acc_corr = ADD_clf_correct_use_corr / AAD_clf_total
    tb_logger.write_training_epoch(AAD_clf_Acc, AAD_clf_Acc_corr, basic_routine, exp)

def test(epoch, exp, tb_logger):
    with torch.no_grad():
        mm_vae = exp.mm_vae;
        mm_vae.eval();
        exp.mm_vae = mm_vae;

        clf = exp.clf_AAD;
        clf.eval();
        exp.clf_AAD = clf;

        d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                            shuffle=True,
                            num_workers=8, drop_last=True);

        ADD_clf_correct, ADD_clf_correct_use_corr, AAD_clf_total = 0, 0, 0;
        mix_audio_correct = 0;
        m0_clf_correct, m1_clf_correct, m2_clf_correct, m0_only_clf_correct = 0, 0, 0, 0;
        m0_clf_Acc, m1_clf_Acc, m2_clf_Acc, m0_only_clf_Acc = 0, 0, 0, 0;
        for iteration, batch in enumerate(d_loader):
            basic_routine = basic_routine_epoch(exp, batch);
            total_loss = basic_routine['total_loss'];

            ADD_clf_correct += basic_routine['ADD_clf_correct']
            ADD_clf_correct_use_corr += basic_routine['ADD_clf_correct_use_corr']
            AAD_clf_total += basic_routine['AAD_clf_total']

            mix_audio_result = mix_audio_epoch(exp, batch)
            mix_audio_correct += mix_audio_result['ADD_clf_correct']

            modality_missing_results = modality_missing_epoch(exp, batch)
            m0_clf_correct += modality_missing_results['ADD_clf_correct']['m0']
            m1_clf_correct += modality_missing_results['ADD_clf_correct']['m1']
            m2_clf_correct += modality_missing_results['ADD_clf_correct']['m2']
            m0_only_clf_correct += modality_missing_results['ADD_clf_correct']['m0_only']

            tb_logger.write_testing_logs(basic_routine);

        # for joint training
        AAD_clf_Acc = ADD_clf_correct / AAD_clf_total
        AAD_clf_Acc_corr = ADD_clf_correct_use_corr / AAD_clf_total

        mix_audio_clf_Acc = mix_audio_correct / AAD_clf_total

        m0_clf_Acc = m0_clf_correct / AAD_clf_total
        m1_clf_Acc = m1_clf_correct / AAD_clf_total
        m2_clf_Acc = m2_clf_correct / AAD_clf_total
        m0_only_clf_Acc = m0_only_clf_correct / AAD_clf_total

        tb_logger.write_testing_epoch(AAD_clf_Acc, AAD_clf_Acc_corr,
                                      [m0_clf_Acc, m1_clf_Acc, m2_clf_Acc, m0_only_clf_Acc], mix_audio_clf_Acc, basic_routine, exp,
                                      modality_missing_results, mix_audio_result)
    return AAD_clf_Acc

def test_sub(epoch, exp, tb_logger):
    with torch.no_grad():
        mm_vae = exp.mm_vae;
        mm_vae.eval();
        exp.mm_vae = mm_vae;

        clf = exp.clf_AAD;
        clf.eval();
        exp.clf_AAD = clf;

        sub_Acc = []
        sub_cossim = []
        for sub_id in range(exp.flags.sub_num):
            test_set = exp.dataset_test_sub[sub_id]
            d_loader = DataLoader(test_set, batch_size=exp.flags.batch_size,
                                shuffle=True,
                                num_workers=8, drop_last=True);

            ADD_clf_correct, AAD_clf_total = 0, 0;
            data_num = 0
            iteration_cossim = []
            for iteration, batch in enumerate(d_loader):
                basic_routine = basic_routine_epoch(exp, batch);
                ADD_clf_correct += basic_routine['ADD_clf_correct']
                AAD_clf_total += basic_routine['AAD_clf_total']
                iteration_cossim.append(basic_routine['cossim'].sum())
                data_num += len(basic_routine['cossim'])
            AAD_clf_Acc = ADD_clf_correct / AAD_clf_total
            sub_Acc.append(AAD_clf_Acc)
            sub_cossim.append(sum(iteration_cossim)/data_num)
        tb_logger.write_testing_epoch_sub(sub_Acc, sub_cossim, basic_routine, exp)

def run_epochs(exp):
    # initialize summary writer
    writer = SummaryWriter(exp.flags.dir_logs)
    tb_logger = TBLogger(exp.flags.str_experiment, writer)
    str_flags = utils.save_and_log_flags(exp.flags);
    tb_logger.writer.add_text('FLAGS', str_flags, 0)

    # if exp.flags.train_clf_only:
    #     state_dict = torch.load(exp.flags.vae_model_path)
    #     exp.mm_vae.load_state_dict(state_dict)


    if not exp.flags.train_clf_only:
        print('Multi-Modal VAE training epochs progress:')
        for epoch in range(exp.flags.start_epoch, exp.flags.end_epoch):
            utils.printProgressBar(epoch, exp.flags.end_epoch)
            # one epoch of training and testing
            train(epoch, exp, tb_logger);
            # debug
            test_sub(epoch, exp, tb_logger)
            AAD_clf_Acc = test(epoch, exp, tb_logger);
            if AAD_clf_Acc > exp.best_clf_Acc:
                # print(AAD_clf_Acc)
                exp.best_clf_Acc = AAD_clf_Acc
                test_sub(epoch, exp, tb_logger)
                dir_network_epoch = os.path.join(exp.flags.dir_checkpoints, str(epoch).zfill(4));
                if not os.path.exists(dir_network_epoch):
                    os.makedirs(dir_network_epoch);
                torch.save(exp.mm_vae.state_dict(),
                           os.path.join(dir_network_epoch, exp.flags.mm_vae_save_name))
                torch.save(exp.clf_AAD.state_dict(),
                           os.path.join(dir_network_epoch, exp.flags.clf_save_name))
            tb_logger.epoch_test += 1

    print('finish！')
