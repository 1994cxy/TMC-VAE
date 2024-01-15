import torch
import torch.nn as nn


class ClfAAD(nn.Module):
    """
    linear classifier
    """
    def __init__(self, flags):
        super(ClfAAD, self).__init__()
        self.fc_dp_prob = flags.clf_dp_prob
        self.flags = flags

        # self.lstm_hidden_size = 48
        # use_bidirectional = True

        # if True == use_bidirectional:
        #     self.num_direction = 2
        #     self.direction_scale = 0.5
        # else:
        #     self.num_direction = 1
        #     self.direction_scale = 1
        #
        # self.lstm1 = nn.LSTM((self.flags.class_dim),
        #                      int(self.lstm_hidden_size * self.direction_scale),
        #                      bidirectional=use_bidirectional, batch_first=True,
        #                      num_layers=3)

        self.clf = nn.Sequential(
            nn.Linear(self.flags.class_dim, 64),
            # nn.Linear(2*int(self.lstm_hidden_size * self.direction_scale), 64),
            nn.Dropout(p=self.fc_dp_prob),
            # nn.Linear(128, 128),
            # nn.Dropout(p=self.fc_dp_prob),
            nn.Linear(64, 32),
            nn.Dropout(p=self.fc_dp_prob),
            nn.Linear(32, 2),
            # nn.Sigmoid()
            nn.Softmax(-1)
        )

    def forward(self, x):
        """

        Args:
            x:  z samples from the multimodal posterior distribution

        Returns:

        """
        # x, (hidden_state, cell_state) = self.lstm1(x)
        # x = x.reshape(-1, 2*self.flags.class_dim)
        x = self.clf(x)

        return x