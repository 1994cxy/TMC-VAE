import torch
import torch.nn as nn

from utils.utils import Flatten, Unflatten


class EncoderAud(nn.Module):
    """

    """
    def __init__(self, flags, AE=False):
        super(EncoderAud, self).__init__()

        self.flags = flags

        # param for conv channel and dropout prob.
        self.num_conv_kernels_1 = 32
        self.num_conv_kernels_2 = 16
        self.num_conv_kernels_3 = 8
        self.num_conv_kernels_4 = 1
        self.aud_dp_prob = flags.aud_dp_prob

        if self.flags.win_length in [3, 2, 1]:
            # input shape (151, 257), 3s win_length
            self.shared_encoder = nn.Sequential(
                    nn.Conv2d(1, self.num_conv_kernels_1, kernel_size=(1, 7), padding=(0, 3)),
                    nn.BatchNorm2d(self.num_conv_kernels_1),
                    nn.ReLU(),
                    nn.Dropout(p=self.aud_dp_prob),
                    # -> (-1, 32, 151, 257)

                    nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7, 1), padding=(0, 0)),
                    nn.MaxPool2d((1, 4)),
                    nn.BatchNorm2d(self.num_conv_kernels_1),
                    nn.ReLU(),
                    nn.Dropout(p=self.aud_dp_prob),
                    # -> (-1, 32, 145, 257)

                    nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(3, 5), padding=(0, 16), dilation=(8, 8)),
                    nn.MaxPool2d((1, 2)),
                    nn.BatchNorm2d(self.num_conv_kernels_1),
                    nn.ReLU(),
                    nn.Dropout(p=self.aud_dp_prob),
                    # -> (-1, 32, 129, 64)

                    nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(3, 3), padding=(0, 16), dilation=(16, 16)),
                    nn.BatchNorm2d(self.num_conv_kernels_1),
                    nn.ReLU(),
                    nn.Dropout(p=self.aud_dp_prob),
                    # -> (-1, 32, 97, 32)

                    nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_4, kernel_size=(1, 1)),
                    nn.MaxPool2d((2, 2)),
                    nn.BatchNorm2d(self.num_conv_kernels_4),
                    nn.ReLU(),
                    nn.Dropout(p=self.aud_dp_prob),
                    # -> (-1, 1, 48, 16)

                    Flatten(),
                    nn.Linear(1*48*16, flags.style_dim + flags.class_dim),       # -> (ndim_private + ndim_shared)
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
        # self.class_mu = nn.Linear(2 * int(self.flags.class_dim * self.direction_scale), flags.class_dim)
        # self.class_logvar = nn.Linear(2 * int(self.flags.class_dim * self.direction_scale), flags.class_dim)

        self.class_mu = nn.Linear(flags.class_dim, flags.class_dim)
        self.class_logvar = nn.Linear(flags.class_dim, flags.class_dim)

    def forward(self, x):
        # todo: the speaker one and speaker two is different modal?
        # h = []
        # for channel in range(x.shape[1]):
        #     h_temp = self.shared_encoder(x[:, channel, :, :].unsqueeze(1))
        #     h.append(h_temp)
        # h = h[0]+h[1]
        # if self.flags.win_length==2:
        #     x = x.repeat(1, 1, 2, 1)
        #     x = x[:, :, :151, :]
        # elif self.flags.win_length==1:
        #     x = x.repeat(1, 1, 3, 1)
        h = self.shared_encoder(x)
        # h, (hidden_state, cell_state) = self.lstm1(h)

        return self.class_mu(h), self.class_logvar(h)


class DecoderAud(nn.Module):
    """

    """
    def __init__(self, flags):
        super(DecoderAud, self).__init__()
        self.flags = flags

        # param for conv channel and dropout prob.
        self.num_conv_kernels_1 = 32
        self.num_conv_kernels_2 = 16
        self.num_conv_kernels_3 = 8
        self.num_conv_kernels_4 = 1
        self.aud_dp_prob = flags.aud_dp_prob

        if self.flags.win_length in [3, 2, 1]:
            self.decoder = nn.Sequential(
                # -> (-1, 1*48*16)
                nn.Linear(flags.style_dim + flags.class_dim, 1 * 48 * 16),
                nn.BatchNorm1d(1 * 48 * 16),
                nn.ReLU(),
                # -> (-1, 1, 48, 16)
                Unflatten((1, 48, 16)),

                nn.ConvTranspose2d(self.num_conv_kernels_4, self.num_conv_kernels_4, kernel_size=(3, 2), stride=(2, 2)),
                nn.BatchNorm2d(self.num_conv_kernels_4),
                nn.ReLU(),
                # -> (-1, 1, 97, 32)

                nn.ConvTranspose2d(self.num_conv_kernels_4, self.num_conv_kernels_1, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(self.num_conv_kernels_1),
                nn.ReLU(),
                # -> (-1, 32, 97, 32)

                nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(5, 2), stride=(1, 2),
                                   dilation=(8, 1)),
                nn.BatchNorm2d(self.num_conv_kernels_1),
                nn.ReLU(),
                # -> (-1, 32, 129, 64)

                nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(5, 2), stride=(1, 2),
                                   dilation=(4, 1)),
                nn.BatchNorm2d(self.num_conv_kernels_1),
                nn.ReLU(),
                # -> (-1, 32, 145, 128)

                nn.ConvTranspose2d(self.num_conv_kernels_1, 1, kernel_size=(4, 3), stride=(1, 2), dilation=(2, 1)),
                # -> (-1, 1, 151, 257)
            )

        # elif self.flags.win_length==2:
        #     self.decoder = nn.Sequential(
        #         # -> (-1, 1*32*16)
        #         nn.Linear(flags.style_dim + flags.class_dim, 1 * 32 * 16),
        #         nn.BatchNorm1d(1 * 32 * 16),
        #         nn.ReLU(),
        #         # -> (-1, 1, 32, 16)
        #         Unflatten((1, 32, 16)),
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_4, self.num_conv_kernels_4, kernel_size=(3, 2), stride=(2, 2)),
        #         nn.BatchNorm2d(self.num_conv_kernels_4),
        #         nn.ReLU(),
        #         # -> (-1, 1, 65, 32)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_4, self.num_conv_kernels_1, kernel_size=(1, 1), stride=(1, 1)),
        #         nn.BatchNorm2d(self.num_conv_kernels_1),
        #         nn.ReLU(),
        #         # -> (-1, 32, 65, 32)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(5, 2), stride=(1, 2),
        #                            dilation=(5, 1)),
        #         nn.BatchNorm2d(self.num_conv_kernels_1),
        #         nn.ReLU(),
        #         # -> (-1, 32, 85, 64)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(6, 2), stride=(1, 2),
        #                            dilation=(2, 1)),
        #         nn.BatchNorm2d(self.num_conv_kernels_1),
        #         nn.ReLU(),
        #         # -> (-1, 32, 95, 128)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_1, 1, kernel_size=(4, 3), stride=(1, 2), dilation=(2, 1)),
        #         # -> (-1, 1, 101, 257)
        #     )
        #
        # elif self.flags.win_length==1:
        #     self.decoder = nn.Sequential(
        #         # -> (-1, 1*16*16)
        #         nn.Linear(flags.style_dim + flags.class_dim, 1 * 16 * 16),
        #         nn.BatchNorm1d(1 * 16 * 16),
        #         nn.ReLU(),
        #         # -> (-1, 1, 16, 16)
        #         Unflatten((1, 16, 16)),
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_4, self.num_conv_kernels_4, kernel_size=(3, 2), stride=(2, 2)),
        #         nn.BatchNorm2d(self.num_conv_kernels_4),
        #         nn.ReLU(),
        #         # -> (-1, 1, 33, 32)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_4, self.num_conv_kernels_1, kernel_size=(1, 1), stride=(1, 1)),
        #         nn.BatchNorm2d(self.num_conv_kernels_1),
        #         nn.ReLU(),
        #         # -> (-1, 32, 33, 32)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(5, 2), stride=(1, 2),
        #                            dilation=(2, 1)),
        #         nn.BatchNorm2d(self.num_conv_kernels_1),
        #         nn.ReLU(),
        #         # -> (-1, 32, 41, 64)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(3, 2), stride=(1, 2),
        #                            dilation=(2, 1)),
        #         nn.BatchNorm2d(self.num_conv_kernels_1),
        #         nn.ReLU(),
        #         # -> (-1, 32, 45, 128)
        #
        #         nn.ConvTranspose2d(self.num_conv_kernels_1, 1, kernel_size=(4, 3), stride=(1, 2), dilation=(2, 1)),
        #         # -> (-1, 1, 51, 257)
        #     )

    def forward(self, style_latent_space, class_latent_space):
        if self.flags.factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat


if __name__=='__main__':
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument('--factorized_representation', type=int, default=0)
    parser.add_argument('--style_dim', type=int, default=0)
    parser.add_argument('--class_dim', type=int, default=64)

    FLAGS = parser.parse_args()

    # win_length = 3s, hop_length=1s, EEG=[192,10], stft_speech=[151, 257]
    # win_length = 2s, hop_length=1s, EEG=[128,10], stft_speech=[101, 257]
    # win_length = 1s, hop_length=0.5s, EEG=[64,10], stft_speech=[51, 257]


    # # 3s win_length
    # FLAGS.win_length = 3  #channel*H*W
    # 2s win_length
    FLAGS.win_length = 2  #channel*H*W
    # 1s win_length
    # FLAGS.win_length = 1  # channel*H*W

    FLAGS.aud_dp_prob = 0.1

    encoder = EncoderAud(FLAGS)
    summary(encoder, input_size=[(1, 151, 257)], device='cpu')

    # decoder = DecoderAud(FLAGS)
    # summary(decoder, input_size=[(1, 64), (1, 64)], device='cpu')



