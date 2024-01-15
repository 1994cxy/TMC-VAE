import torch
import torch.nn as nn
import math
from utils.utils import Flatten, Unflatten, cal_conv_output_size, cal_deconv_output_size, cal_deconv_para

def cal_eeg_channel_output(eeg_channel):
    output_size = cal_conv_output_size(eeg_channel, 1, 1, 0)

    output_size_1 = math.floor(output_size) / 2
    output_size_2 = cal_conv_output_size(math.floor(output_size_1), 5, 1, 2)
    output_size_2 = math.floor(output_size_2) / 5

    return int(output_size_2), int(output_size_1)

class EncoderEEG(nn.Module):
    """

    """
    def __init__(self, flags, AE=False):
        super(EncoderEEG, self).__init__()

        self.flags = flags

        # param for conv channel and dropout prob.
        self.num_conv_kernels_1 = 32
        self.num_conv_kernels_2 = 16
        self.num_conv_kernels_3 = 8
        self.num_conv_kernels_4 = 1
        self.E_dp_prob = flags.eeg_dp_prob

        self.cnn_output_t = 16 * 3

        self.eeg_channel_output, _ = cal_eeg_channel_output(self.flags.eeg_channel)

        # input shape (192, 10)
        self.shared_encoder = nn.Sequential(
                # -> (-1, 32, 193, 10)
                nn.Conv2d(self.flags.eeg_band, self.num_conv_kernels_1, kernel_size=(24, 1), padding=(12, 0)),
                nn.MaxPool2d((2, 1)),
                nn.BatchNorm2d(self.num_conv_kernels_1),
                nn.ReLU(),
                nn.Dropout(p=self.E_dp_prob),
                # -> (-1, 32, 96, 10)

                nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7, 1), padding=(6, 0), dilation=(2, 1)),
                nn.MaxPool2d((1, 2)),
                nn.BatchNorm2d(self.num_conv_kernels_1),
                nn.ReLU(),
                nn.Dropout(p=self.E_dp_prob),
                # -> (-1, 32, 96, 5)

                nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7, 5), padding=(3, 2)),
                nn.MaxPool2d((2, 5)),
                nn.BatchNorm2d(self.num_conv_kernels_1),
                nn.ReLU(),
                nn.Dropout(p=self.E_dp_prob),
                # -> (-1, 32, 48, 1)
                # -> (-1, 32, 48, 6) for 64 channel EEG data

                nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7, 1), padding=(3, 0)),
                nn.BatchNorm2d(self.num_conv_kernels_1),
                nn.ReLU(),
                nn.Dropout(p=self.E_dp_prob),

                Flatten(),
                nn.Linear(self.num_conv_kernels_1*self.cnn_output_t*self.eeg_channel_output, flags.style_dim + flags.class_dim),       # -> (ndim_private + ndim_shared)
                nn.ReLU(),
        )

        # use_bidirectional = True
        #
        # if True == use_bidirectional:
        #     self.num_direction = 2
        #     self.direction_scale = 0.5
        # else:
        #     self.num_direction = 1
        #     self.direction_scale = 1
        #
        # self.lstm1 = nn.LSTM((self.flags.class_dim),
        #                      int(self.flags.class_dim * self.direction_scale),
        #                      bidirectional=use_bidirectional, batch_first=True)
        #
        # self.class_mu = nn.Linear(2*int(self.flags.class_dim * self.direction_scale), flags.class_dim)
        # self.class_logvar = nn.Linear(2*int(self.flags.class_dim * self.direction_scale), flags.class_dim)

        self.class_mu = nn.Linear(flags.class_dim, flags.class_dim)
        self.class_logvar = nn.Linear(flags.class_dim, flags.class_dim)

    def forward(self, x):
        # if self.flags.win_length == 2:
        #     x = x.repeat(1, 1, 2, 1)
        #     x = x[:, :, :192, :]
        # elif self.flags.win_length == 1:
        #     x = x.repeat(1, 1, 3, 1)
        h = self.shared_encoder(x)
        # h, (hidden_state, cell_state) = self.lstm1(h)

        return self.class_mu(h), self.class_logvar(h)


class DecoderEEG(nn.Module):
    """

    """
    def __init__(self, flags):
        super(DecoderEEG, self).__init__()
        self.flags = flags

        # param for conv channel and dropout prob.
        self.num_conv_kernels_1 = 32
        self.num_conv_kernels_2 = 16
        self.num_conv_kernels_3 = 8
        self.num_conv_kernels_4 = 1
        self.E_dp_prob = flags.eeg_dp_prob

        self.cnn_output_t = 16 * 3

        self.eeg_channel_output, self.eeg_channel_output_mid = cal_eeg_channel_output(self.flags.eeg_channel)

        filter_size_1, stride_1 = cal_deconv_para(self.eeg_channel_output_mid, self.eeg_channel_output)
        filter_size_2, stride_2 = cal_deconv_para(self.flags.eeg_channel, self.eeg_channel_output_mid)

        self.decoder = nn.Sequential(
            # -> (-1, 32*48*1)
            nn.Linear(flags.style_dim + flags.class_dim,
                      self.num_conv_kernels_1 * self.cnn_output_t * self.eeg_channel_output),
            nn.BatchNorm1d(self.num_conv_kernels_1 * self.cnn_output_t * self.eeg_channel_output),
            nn.ReLU(),
            # -> (-1, 32, 48, 1)
            Unflatten((self.num_conv_kernels_1, self.cnn_output_t, self.eeg_channel_output)),

            nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(2, filter_size_1),
                               stride=(2, stride_1)),
            nn.BatchNorm2d(self.num_conv_kernels_1),
            nn.ReLU(),
            # -> (-1, 32, 96, 5)

            nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(1, filter_size_2),
                               stride=(1, stride_2)),
            nn.BatchNorm2d(self.num_conv_kernels_1),
            nn.ReLU(),
            # -> (-1, 32, 96, 10)

            nn.ConvTranspose2d(self.num_conv_kernels_1, self.flags.eeg_band, kernel_size=(2, 1), stride=(2, 1)),
            # -> (-1, 1, 192, 10)
        )

    def forward(self, style_latent_space, class_latent_space):
        if self.flags.factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        return x_hat


if __name__=='__main__':
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument('--factorized_representation', type=int, default=0)
    parser.add_argument('--style_dim', type=int, default=0)
    parser.add_argument('--class_dim', type=int, default=64)

    FLAGS = parser.parse_args()

    # win_lenth = 3s, hop_lenth=1s, EEG=[192,10], speech=[151, 257]
    # win_lenth = 2s, hop_lenth=1s, EEG=[128,10], speech=[101, 257]
    # win_lenth = 1s, hop_lenth=0.5s, EEG=[64,10], speech=[51, 257]

    # 3s win_length
    FLAGS.win_length = 3  #channel*H*W
    # 2s win_length
    # FLAGS.win_length = 2  #channel*H*W
    # 1s win_length
    # FLAGS.win_length = 1  # channel*H*W

    FLAGS.eeg_dp_prob = 0.1

    # encoder = EncoderEEG(FLAGS)
    # summary(encoder, input_size=[(1, 192, 64)], device='cpu')

    decoder = DecoderEEG(FLAGS)
    summary(decoder, input_size=[(1, 64), (1, 64)], device='cpu')



