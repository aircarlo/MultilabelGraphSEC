import os
import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from torch.distributions.beta import Beta
from audio_parser import MELspParser
import soundfile as sf
import numpy as np


class FSD50K_train_dataset(Dataset):
    """
    Chunk-level dataset.
    """
    def __init__(self, par):
        super(FSD50K_train_dataset, self).__init__()
        self.labels_map_path = os.path.join(par['meta_root'], par['labels_map'])
        with open(self.labels_map_path, 'r') as fd:
            self.labels_map = json.load(fd)
        if par['merge_train_val']:
            df_train = pd.read_csv(os.path.join(par['meta_root'], par['train_manifest']))
            df_val = pd.read_csv(os.path.join(par['meta_root'], par['val_manifest']))
            df = pd.merge(df_train, df_val, how='outer')
        else:
            df = pd.read_csv(os.path.join(par['meta_root'], par['train_manifest']))

        self.files = df['files'].values
        self.labels = df['labels'].values
        assert len(self.files) == len(self.labels)
        self.len = len(self.files)

        self.sr = par['sample_rate']
        self.n_fft = par['n_fft']
        self.win_len = par['win_len']
        self.hop_len = par['hop_len']
        self.n_mels = par['n_mels']
        self.mixup = par['mixup']  # whether to perform mixup
        self.spec_augm = par['spec_augm']  # whether to perform spectrogram t-f masking
        self.beta = Beta(torch.tensor([0.2]), torch.tensor([0.2]))  # beta distribution for mixup param

        self.spec_parser = MELspParser(sr=self.sr,
                                       n_fft=self.n_fft,
                                       win_length=self.win_len,
                                       hop_length=self.hop_len,
                                       n_mels=self.n_mels)

    def __getitem__(self, index):
        if self.mixup and index > self.len - 1:
            chunk_melsp, label_tensor = self.__mixup__(index - self.len)
        elif self.spec_augm and index > self.len - 1:
            chunk_melsp, label_tensor = self.__spec_augment__(index - self.len)
        else:
            f = self.files[index]
            lbls = self.labels[index]
            label_tensor = self.__parse_labels__(lbls)
            chunk_audio = sf.read(f)[0].astype('float32')
            chunk_melsp = torch.from_numpy(self.spec_parser(chunk_audio))

        return chunk_melsp, label_tensor

    def __parse_labels__(self, lbls: str) -> torch.Tensor:  # from list of string labels, to one-hot-encoded vector
        label_tensor = torch.zeros(len(self.labels_map)).float()
        for lbl in lbls.split(','):
            label_tensor[self.labels_map[lbl]] = 1
        return label_tensor

    def __mixup__(self, index):
        new_index = torch.randint(0, self.len - 1, (1,))  # sample a new index
        # extract 1st chunk of mixup
        f_1 = self.files[index]
        lbls_1 = self.labels[index]
        label_tensor_1 = self.__parse_labels__(lbls_1)
        chunk_audio_1 = sf.read(f_1)[0].astype('float32')
        chunk_melsp_1 = torch.from_numpy(self.spec_parser(chunk_audio_1))
        # extract 2nd chunk of mixup
        f_2 = self.files[new_index]
        lbls_2 = self.labels[new_index]
        label_tensor_2 = self.__parse_labels__(lbls_2)
        chunk_audio_2 = sf.read(f_2)[0].astype('float32')
        chunk_melsp_2 = torch.from_numpy(self.spec_parser(chunk_audio_2))
        # Mixup the spectrogram and labels accordingly
        lam = self.beta.sample()
        chunk_mixup = torch.mul(chunk_melsp_1, lam) + torch.mul(chunk_melsp_2, (1 - lam))
        label_mixup = torch.mul(label_tensor_1, lam) + torch.mul(label_tensor_2, (1 - lam))

        return chunk_mixup, label_mixup

    def __spec_augment__(self, index):
        # TODO: parameterize params
        num_mask = 2
        freq_masking_max_percentage = 0.15
        time_masking_max_percentage = 0.3
        # extract chunk to mask
        f = self.files[index]
        lbls = self.labels[index]
        label_tensor = self.__parse_labels__(lbls)
        chunk_audio = sf.read(f)[0].astype('float32')
        chunk_melsp = torch.from_numpy(self.spec_parser(chunk_audio))
        for i in range(num_mask):
            all_frames_num, all_freqs_num = chunk_melsp.shape
            freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            chunk_melsp[:, f0:f0 + num_freqs_to_mask] = 0
            time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            chunk_melsp[t0:t0 + num_frames_to_mask, :] = 0

        return chunk_melsp, label_tensor

    def __len__(self):
        if self.mixup or self.spec_augm:
            return 2*self.len  # double the ids to obtain 50% augmentation
        else:
            return self.len


class FSD50K_test_dataset(Dataset):
    """
    Clip-level dataset.
    """
    def __init__(self, par):
        super(FSD50K_test_dataset, self).__init__()
        self.labels_map_path = os.path.join(par['meta_root'], par['labels_map'])
        with open(self.labels_map_path, 'r') as fd:
            self.labels_map = json.load(fd)
        df = pd.read_csv(os.path.join(par['meta_root'], par['test_manifest']))
        self.files = df['files'].values
        self.labels = df['labels'].values
        self.exts = df['ext'].values
        self.unique_exts = np.unique(self.exts)
        assert len(self.files) == len(self.labels) == len(self.exts)
        self.len = len(self.unique_exts)  # works at clip-level!!
        print(f'Lenght of test set: {self.len} files')

        self.sr = par['sample_rate']
        self.n_fft = par['n_fft']
        self.win_len = par['win_len']
        self.hop_len = par['hop_len']
        self.n_mels = par['n_mels']

        self.spec_parser = MELspParser(sr=self.sr,
                                       n_fft=self.n_fft,
                                       win_length=self.win_len,
                                       hop_length=self.hop_len,
                                       n_mels=self.n_mels)

    def __getitem__(self, index):
        tgt_ext = self.unique_exts[index]
        idxs = np.where(self.exts == tgt_ext)[0]
        tensors = []
        label_tensors = []
        for idx in idxs:
            f = self.files[idx]
            lbls = self.labels[idx]
            label_tensor = self.__parse_labels__(lbls)
            chunk_audio = sf.read(f)[0].astype('float32')
            chunk_melsp = torch.from_numpy(self.spec_parser(chunk_audio))
            tensors.append(chunk_melsp.unsqueeze(0).unsqueeze(0))
            label_tensors.append(label_tensor.unsqueeze(0))
        tensors = torch.cat(tensors)
        return tensors, label_tensors[0]

    def __parse_labels__(self, lbls: str) -> torch.Tensor:  # from list of string labels, to one-hot-encoded vector
        label_tensor = torch.zeros(len(self.labels_map)).float()
        for lbl in lbls.split(','):
            label_tensor[self.labels_map[lbl]] = 1
        return label_tensor

    def __len__(self):
        return self.len
    
