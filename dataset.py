"""
    Training dataset of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/blob/master/s3prl/pretrain/bucket_dataset.py)
    Reference author: Andy T. Liu (https://github.com/andi611)
"""
import numpy as np
import torch
import os
import random
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

class FeatLabelDataset(Dataset):
    def __init__(self, data_config):
        super(FeatLabelDataset, self).__init__()
        self.bucket_size = data_config['train_batch_size']
        self.sets = data_config['sets']
        self.max_timestep = data_config['max_timestep']
        self.sample_length = data_config['sequence_length']
        if self.sample_length > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_length)

        # You can assign multiple .csv data files in sets
        tables = [pd.read_csv(s) for s in self.sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        print('[Dataset] - Training data from these sets:', str(self.sets))

        # Drop seqs that are too long
        if self.max_timestep > 0:
            self.table = self.table[self.table.length < self.max_timestep]
        # Drop seqs that are too short
        if self.max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * self.max_timestep)]

        X = self.table['file_path'].tolist()
        Y = self.table['label_path'].tolist()
        X_lens = self.table['length'].tolist()
        self.num_samples = len(X)
        print('[Dataset] - Number of individual training instances:', self.num_samples)

        # Use bucketing to make utterances with closest length in a batch
        self.X = []
        self.Y = []
        batch_x, batch_y, batch_len = [], [], []
        for x, y, x_len in zip(X, Y, X_lens):
            batch_x.append(x)
            batch_y.append(y)
            batch_len.append(x_len)
            # Fill in batch_x until batch is full
            if len(batch_x) == self.bucket_size:
                self.X.append(batch_x)
                self.Y.append(batch_y)
                batch_x, batch_y, batch_len = [], [], []
        
        # Gather the last batch
        if len(batch_x) > 1: 
            self.X.append(batch_x)
            self.Y.append(batch_y)

    def _sample(self, x, y):
        if self.sample_length <= 0: return x, y
        if len(x) < self.sample_length: return x, y
        idx = random.randint(0, len(x)-self.sample_length)
        return x[idx:idx+self.sample_length], y[idx:idx+self.sample_length]

    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        # Hack bucketing
        items = items[0] 
        return items

class MelFeatDataset(FeatLabelDataset):
    
    def __init__(self, frame_period, data_config):
        super(MelFeatDataset, self).__init__(data_config)
        self.frame_period = frame_period

    def _load_feat(self, feat_path):
        feat = np.load(feat_path)
        if self.frame_period == 20:
            # Concatenating every two contiguous frame to double the frame rate to 20 ms
            odd_feat = feat[::2,:]
            even_feat = feat[1::2,:]
            if odd_feat.shape[0] != even_feat.shape[0]:
                even_feat = np.concatenate((even_feat, np.zeros((1,even_feat.shape[1]))), axis=0)
            feat = np.concatenate((odd_feat, even_feat), axis=1)
        return torch.FloatTensor(feat)

    def _load_label(self, label_path, feat_len):
        label = np.load(label_path)
        label_len = label.shape[0]
        if self.frame_period == 20 and feat_len != label_len:
            label = label[::2]
        return torch.LongTensor(label)

    def __getitem__(self, index):
        # Load acoustic feature, label and pad
        x_batch, y_batch = [], []
        for x_file, y_file in zip(self.X[index], self.Y[index]):
            feat = self._load_feat(x_file)
            label = self._load_label(y_file, feat.shape[0])
            x, y = self._sample(feat, label)
            x_batch.append(x)
            y_batch.append(y)

        x_len = [len(x_b) for x_b in x_batch]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        y_pad_batch = pad_sequence(y_batch, batch_first=True) 

        pad_mask = torch.ones(x_pad_batch.shape[:-1])
        # Zero vectors for padding dimension
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_len[idx]:] = 0

        return x_pad_batch, y_pad_batch, pad_mask, x_len

