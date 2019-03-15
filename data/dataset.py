import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect


class NpssDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 item_length,
                 target_length,
                 train=True):

        #           |----receptive_field----|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                             | | | | | | | | |
        self.dataset_file = dataset_file
        self._item_length = item_length
        self.target_length = target_length

        self.data = np.load(self.dataset_file)
        self.data = self.data / 128

        self._length = 0
        self.calculate_length()
        self.train = train
        print("one hot input")


    def calculate_length(self):

        available_length = self.data.shape[0] - (self._item_length - (self.target_length - 1)) - 1
        self._length = math.floor(available_length / self.target_length)


    def __getitem__(self, idx):

        sample_index = idx * self.target_length

        sample = self.data[sample_index:sample_index+self._item_length+1, :]

        example = torch.from_numpy(sample)

        item = example[:self._item_length].transpose(0, 1)
        target = example[-self.target_length:].transpose(0, 1)
        return item, target

    def __len__(self):

        return self._length


