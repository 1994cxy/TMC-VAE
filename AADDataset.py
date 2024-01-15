import numpy as np
import argparse
import torch
import os
import glob

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
from scipy import stats, signal


class AADDataset(Dataset):
    def __init__(self, unimodal_datapaths, csv_path, flags, transform=None, target_transform=None, sub_id=None):
        """
                    Args: unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                            correspond by index. Therefore the numbers of samples of all datapaths should match.
                        transform: tranforms on colored MNIST digits.
                        target_transform: transforms on labels.
                """
        super().__init__()
        self.flags = flags
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = transform
        self.target_transform = target_transform

        self.csv_path = csv_path
        self.root_path = '/'.join(csv_path.split('/')[:-1])
        self.get_data(sub_id)

        # band pass filtering (1-9 Hz)
        self.sos_bp = ([[0.00432289, 0.00390431, 0.00432289, 1., -0.68058302, 0.18722999],
                        [1., -0.23389564, 1., 1., -0.8336818, 0.39304264],
                        [1., -0.73839576, 1., 1., -1.02270308, 0.62849011],
                        [1., 0., -1., 1., -1.20601611, 0.27919438],
                        [1., -1.99869015, 1., 1., -1.81621689, 0.82786496],
                        [1., -0.9224206, 1., 1., -1.20537201, 0.86885965],
                        [1., -1.99561804, 1., 1., -1.86732577, 0.8802607],
                        [1., -1.99248629, 1., 1., -1.91594458, 0.93054023],
                        [1., -1.99061636, 1., 1., -1.96125717, 0.97707891]])

        # high pass filtering (1 Hz)
        self.sos_hp = ([[0.7648476, -0.7648476, 0., 1., -0.91094815, 0.],
                        [1., -1.99887104, 1., 1., -1.83030101, 0.83930608],
                        [1., -1.99601526, 1., 1., -1.85531953, 0.86711526],
                        [1., -1.99277274, 1., 1., -1.89612558, 0.91128566],
                        [1., -1.99065917, 1., 1., -1.95060444, 0.96828771]])

        # # save all paths to individual files
        # self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        # for dp in unimodal_datapaths:
        #     files = glob.glob(os.path.join(dp, "*.npy"))
        #     self.file_paths[dp] = files
        # # load dataset partition
        # self.df_data = pd.read_csv(csv_path, header=None)

    def get_data(self, sub_id):
        tmp_filename = os.path.join(self.csv_path)
        df = pd.read_csv(tmp_filename, header=None)
        # df = df[:500]
        if sub_id!=None:
            df.columns = ['file_name', 'label']
            df_sub = df[df['file_name'].str.contains(f'S{sub_id}_')]
            data_list = df_sub.values.tolist()
        else:
            data_list = df.values.tolist()

        Data = []
        for data in data_list:
            labels = data[1]
            eeg_file = data[0]

            # for KUL dataset
            if 'KUL' in self.csv_path:
                audio_file = '_'.join([data[0].split('_')[2], data[0].split('_')[4]])

            # for DTU dataset
            elif 'DTU' in self.csv_path:
                audio_file = '_'.join([data[0].split('_')[0], data[0].split('_')[2]])

            # # create positive and negative data
            # Data.append((eeg_file, audio_file+'_0', int(int(labels)==0)))
            # Data.append((eeg_file, audio_file+'_1', int(int(labels)==1)))

            # just let modal_1=left channel audio and modal_2=right channel audio and label=channel number
            Data.append((eeg_file, audio_file + '_0', audio_file + '_1', labels))

        self.data = Data
        self.num_files = len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        data = self.data[index]

        EEG_data = np.load(os.path.normpath(os.path.join(self.unimodal_datapaths[0], data[0]+'.npy')))
        # EEG_data = signal.sosfiltfilt(self.sos_hp, EEG_data, axis=0)  # a high pass filter with a cut off frequency of 1Hz.
        EEG_data = stats.zscore(EEG_data, axis=0)
        EEG_data = torch.from_numpy(EEG_data.copy()).float()

        # stft audio
        Audio_filename = '_'.join(data[1].split('_')[:-1]) + '.npy'
        Audio_data = np.load(os.path.join(self.unimodal_datapaths[1], Audio_filename))
        Audio_data_0 = torch.from_numpy(Audio_data[0, :, :]).float()
        Audio_data_1 = torch.from_numpy(Audio_data[1, :, :]).float()

        label = data[3]
        label = torch.tensor(label)

        # # origin audio
        # Audio_wav = np.load(os.path.join(self.root_path+'/Speech', Audio_filename))
        # Audio_ori = torch.from_numpy(Audio_wav).float()

        # repeat to 3s
        if self.flags.win_length==2:
            EEG_data = EEG_data.repeat(1, 3, 1)
            EEG_data = EEG_data[:, :192, :]

            Audio_data_0 = Audio_data_0.repeat(3, 1)
            Audio_data_0 = Audio_data_0[:151, :]
            Audio_data_1 = Audio_data_1.repeat(3, 1)
            Audio_data_1 = Audio_data_1[:151, :]

            # Audio_ori = Audio_ori.repeat(1, 3)  # [speaker, fs*win_length]
            # Audio_ori = Audio_ori[:, :16000*3]

        if self.flags.win_length==1:
            EEG_data = EEG_data.repeat(1, 3, 1)
            EEG_data = EEG_data[:, :192, :]

            Audio_data_0 = Audio_data_0.repeat(3, 1)
            Audio_data_0 = Audio_data_0[:151, :]
            Audio_data_1 = Audio_data_1.repeat(3, 1)
            Audio_data_1 = Audio_data_1[:151, :]

            # Audio_ori = Audio_ori.repeat(1, 3)  # [speaker, fs*win_length]
            # Audio_ori = Audio_ori[:, :16000 * 3]

        data_dict = {
            "m0": EEG_data,
            "m1": Audio_data_0.unsqueeze(0),
            "m2": Audio_data_1.unsqueeze(0)
                     }

        return data_dict, label, torch.zeros_like(Audio_data_0)

    def __len__(self):
        return self.num_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-modalities', type=int, default=5)
    parser.add_argument('--savepath-train', type=str, required=True)
    parser.add_argument('--savepath-test', type=str, required=True)
    parser.add_argument('--backgroundimagepath', type=str, required=True)
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    # create dataset
    MMNISTDataset._create_mmnist_dataset(args.savepath_train, args.backgroundimagepath, args.num_modalities, train=True)
    MMNISTDataset._create_mmnist_dataset(args.savepath_test, args.backgroundimagepath, args.num_modalities, train=False)
    print("Done.")