
from .utils import t_sne_visualize, t_sne_visualize_target

class TBLogger():
    def __init__(self, name, writer):
        self.name = name;
        self.writer = writer;
        self.training_prefix = 'train';
        self.testing_prefix = 'val';
        self.clf_training_prefix = 'clf_train'
        self.clf_testing_prefix = 'clf_val'
        self.step_train = 0;
        self.step_test = 0;
        self.epoch_train = 0;
        self.epoch_test = 0;

    def visualize(self, basic_routine, prefix, epoch, exp):
        z_compelete = basic_routine['result']['latent_complete']
        z_target = basic_routine['result']['latent_target']
        label = basic_routine['label']
        # tsne
        tsne_fig = t_sne_visualize(z_target.detach().cpu().numpy(), z_compelete.detach().cpu().numpy(), exp, epoch, prefix, label.cpu().numpy())
        # tsne_fig = t_sne_visualize_target(z_target.detach().cpu().numpy(), exp, epoch, prefix, label.cpu().numpy())
        self.writer.add_figure(f'{prefix}_latent_{epoch}', figure=tsne_fig)
        del tsne_fig

    def visualize_single(self, basic_routine, prefix, epoch, exp):
        if prefix=='mix_audio':
            z_compelete = basic_routine['AAD_clf_z']
        else:
            z_compelete = basic_routine['AAD_clf_z'][prefix]
        label = basic_routine['label']
        # tsne
        tsne_fig = t_sne_visualize_target(z_compelete.detach().cpu().numpy(), exp, epoch, prefix, label.cpu().numpy())
        # tsne_fig = t_sne_visualize_target(z_target.detach().cpu().numpy(), exp, epoch, prefix, label.cpu().numpy())
        self.writer.add_figure(f'{prefix}_mod_miss_latent_{epoch}', figure=tsne_fig)
        del tsne_fig

    def write_subset_div(self, name, result, step):
        latents = result['latents']
        subsets = latents['subsets']
        if len(result['individual_divs']) == 3:
            subsets = ['m_0', 'm_1', 'm_2']
            for k, key in enumerate(subsets):
                self.writer.add_scalars('%s/KLDs' % name,
                                        {key: result['individual_divs'][k]},
                                        step)
        elif len(result['individual_divs']) == 1:
            subsets = ['m0_m1_m2']
            for k, key in enumerate(subsets):
                self.writer.add_scalars('%s/KLDs' % name,
                                        {key: result['individual_divs'][k]},
                                        step)
        elif len(result['individual_divs']) == 4:
            subsets = ['m0', 'm_1', 'm_2', 'm0_m1_m2']
            for k, key in enumerate(subsets):
                self.writer.add_scalars('%s/KLDs' % name,
                                        {key: result['individual_divs'][k]},
                                        step)
        else:
            for k, key in enumerate(subsets.keys()):
                if not result['individual_divs'][k] is None:
                    self.writer.add_scalars('%s/KLDs' % name,
                                            {key: result['individual_divs'][k]},
                                            step)

    def add_basic_logs(self, name, basic_routine, step):
        self.writer.add_scalars('%s/total_loss' % name,
                                {'total_loss': basic_routine['total_loss']},
                                step)
        self.writer.add_scalars('%s/loss_rec' % name,
                                {'loss_rec': basic_routine['loss_rec']},
                                step)
        self.writer.add_scalars('%s/AAD_clf_loss' % name,
                                {'AAD_clf_loss': basic_routine['AAD_clf_loss']},
                                step)
        self.writer.add_scalars('%s/loss_contrastive' % name,
                                {'loss_contrastive': basic_routine['loss_contrastive']},
                                step)
        self.write_subset_div(name, basic_routine['result'], step)
        self.writer.add_scalars('%s/mean_Pearson_corr' % name,
                                {'mean_Pearson_corr': basic_routine['mean_Pearson_corr']},
                                step)
    def write_training_epoch(self, AAD_clf_Acc, AAD_clf_Acc_corr, basic_routine, exp):
        self.writer.add_scalars('%s/AAD_clf_Acc' % self.clf_training_prefix,
                                {'Acc': AAD_clf_Acc},
                                self.epoch_train)
        self.writer.add_scalars('%s/AAD_clf_Acc_corr' % self.clf_training_prefix,
                                {'Acc': AAD_clf_Acc_corr},
                                self.epoch_train)
        self.writer.add_scalars('%s/rec_att_audio_corr' % self.clf_training_prefix,
                                {'att_audio_corr': basic_routine['rec_att_audio_corr']},
                                self.epoch_train)
        self.visualize(basic_routine, self.clf_training_prefix, self.epoch_train, exp)
        self.epoch_train += 1

    def write_testing_epoch(self, AAD_clf_Acc, AAD_clf_Acc_corr, mod_miss_Acc, mix_audio_clf_Acc, basic_routine, exp,
                            modality_missing_results, mix_audio_result):
        self.writer.add_scalars('%s/AAD_clf_Acc_mean' % self.clf_testing_prefix,
                                {'Acc': AAD_clf_Acc},
                                self.epoch_test)
        self.writer.add_scalars('%s/AAD_clf_Acc_mix_audio' % self.clf_testing_prefix,
                                {'mix_audio': mix_audio_clf_Acc},
                                self.epoch_test)
        self.writer.add_scalars('%s/AAD_clf_Acc_mod_miss' % self.clf_testing_prefix,
                                {'m0_miss': mod_miss_Acc[0],
                                 'm1_miss': mod_miss_Acc[1],
                                 'm2_miss': mod_miss_Acc[2],
                                 'm0_only': mod_miss_Acc[3]},
                                self.epoch_test)
        self.writer.add_scalars('%s/AAD_clf_Acc_corr_mean' % self.clf_testing_prefix,
                                {'Acc': AAD_clf_Acc_corr},
                                self.epoch_test)
        self.writer.add_scalars('%s/rec_att_audio_corr' % self.clf_testing_prefix,
                                {'att_audio_corr': basic_routine['rec_att_audio_corr']},
                                self.epoch_test)
        self.visualize_single(mix_audio_result, 'mix_audio', self.epoch_train, exp)
        for prefix in ['m0', 'm1', 'm2', 'm0_only']:
            self.visualize_single(modality_missing_results, prefix, self.epoch_train, exp)
        self.visualize(basic_routine, self.clf_testing_prefix, self.epoch_train, exp)
        # self.epoch_test += 1

    def write_testing_epoch_sub(self, AAD_clf_Acc_sub, sub_cossim, basic_routine, exp):
        for sub_idx, sub_Acc in enumerate(AAD_clf_Acc_sub):
            self.writer.add_scalars('%s/AAD_clf_Acc' % self.clf_testing_prefix,
                                    {f"sub_{sub_idx}_Acc": sub_Acc},
                                    self.epoch_test)
        for sub_idx, cossim in enumerate(sub_cossim):
            self.writer.add_scalars('%s/AAD_cosine_sim' % self.clf_testing_prefix,
                                    {f"sub_{sub_idx}_cosine_sim": cossim},
                                    self.epoch_test)
        self.writer.add_scalars('%s/AAD_cosine_sim_mean' % self.clf_testing_prefix,
                                {'mean_{sub_idx}_cosine_sim': sum(sub_cossim)/len(sub_cossim)},
                                self.epoch_test)

    def write_training_logs(self, basic_routine):
        self.add_basic_logs(self.training_prefix, basic_routine, self.step_train);
        self.step_train += 1;


    def write_testing_logs(self, basic_routine):
        self.add_basic_logs(self.testing_prefix, basic_routine, self.step_test);
        self.step_test += 1;





