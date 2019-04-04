import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
#from torch.distributions.normal import Normal
import librosa as lr
import bisect


class NpssDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 condition_file,
                 receptive_field,
                 target_length,
                 train=True):

        #           |----receptive_field----|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                             | | | | | | | | |
        self.dataset_file = dataset_file
        self._receptive_field = receptive_field
        self.target_length = target_length
        self.item_length = self._receptive_field+self.target_length

        self.data = np.load(self.dataset_file)


        self.conditon = np.load(condition_file).astype(np.float)


        self._length = 0
        self.calculate_length()
        self.train = train



    def calculate_length(self):

        available_length = self.data.shape[0] - self._receptive_field
        self._length = math.floor(available_length / self.target_length)


    def __getitem__(self, idx):

        sample_index = idx * self.target_length

        sample = self.data[sample_index:sample_index+self.item_length, :]

        item_condition = np.transpose(self.conditon[sample_index+self.item_length-1:sample_index+self.item_length, :])

        example = torch.from_numpy(sample)

        item = example[:self._receptive_field].transpose(0, 1)
        target = example[-self.target_length:].transpose(0, 1)
        return (item, item_condition), target

    def __len__(self):

        return self._length


class TimbreDataset(torch.utils.data.Dataset):
    # type 0:harmonic, 1:aperiodic,  2:vuv
    def __init__(self,
                 data_folder,
                 receptive_field,
                 type = 0,
                 target_length=1,
                 train=True):

        #           |----receptive_field----|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                             |
        self.type = type
        self._receptive_field = receptive_field
        self.target_length = target_length
        self.item_length = self._receptive_field+self.target_length

        if train:
            data_folder = os.path.join(data_folder, 'train')
        else:
            data_folder = os.path.join(data_folder, 'test')

        sp_folder = os.path.join(data_folder, 'sp')
        ap_folder = os.path.join(data_folder, 'ap')
        condi_folder = os.path.join(data_folder, 'condition')
        vuv_folder = os.path.join(data_folder, 'vuv')

        # store every data length
        self.data_lengths = []
        self.dataset_files = []
        dirlist = os.listdir(sp_folder)
        for item in dirlist:
            name = item.replace('_sp.npy','')

            sp = np.load(os.path.join(sp_folder, item))
            ap = np.load(os.path.join(ap_folder, name+'_ap.npy'))
            vuv = np.load(os.path.join(vuv_folder, name+'_vuv.npy')).astype(np.uint8)
            condition = np.load(os.path.join(condi_folder, name+'_condi.npy')).astype(np.float)

            assert len(sp) == len(ap) == len(vuv) == len(condition)
            self.data_lengths.append(len(sp))

            # pad zeros(_receptive_field, 60) ahead for each data
            sp = np.pad(sp, ((self._receptive_field, 0), (0, 0)), 'constant', constant_values=0)
            ap = np.pad(ap, ((self._receptive_field, 0), (0, 0)), 'constant', constant_values=0)
            vuv = np.pad(vuv, (self._receptive_field, 0), 'constant', constant_values=0)

            self.dataset_files.append((sp, ap, vuv, condition))
            # for test
            # break

        self._length = 0
        self.calculate_length()
        self.train = train



    def calculate_length(self):

        self._length = 0
        for _len in self.data_lengths:
            available_length = _len
            self._length += math.floor(available_length / self.target_length)


    def __getitem__(self, idx):

        # find witch file it require
        current_files = None
        current_files_idx = 0
        total_len = 0
        for fid, _len in enumerate(self.data_lengths):
            current_files_idx = idx - total_len
            total_len += _len
            if idx < total_len:
                current_files = self.dataset_files[fid]
                break

        sp, ap, vuv, condition = current_files
        target_index = current_files_idx * self.target_length
        item_condition = torch.Tensor(condition[target_index]).unsqueeze(1)

        # notice we pad _receptive_field before so
        sp_sample = torch.Tensor(sp[target_index:target_index+self.item_length, :]).transpose(0, 1)
        sp_item = sp_sample[:, :self._receptive_field]
        sp_target = sp_sample[:, -self.target_length:]

        ap_sample = torch.Tensor(ap[target_index:target_index + self.item_length, :]).transpose(0, 1)
        ap_item = ap_sample[:, :self._receptive_field]
        ap_item = torch.cat((ap_item, sp_item), 0)
        ap_target = ap_sample[:, -self.target_length:]


        vuv_sample = torch.Tensor(vuv[target_index:target_index + self.item_length])
        vuv_item = vuv_sample[:self._receptive_field]
        # notice here ap_item == (ap_item, sp_item) so we dont cat sp item any more
        vuv_item = torch.cat((vuv_item.unsqueeze(0), ap_item), 0)
        vuv_target = vuv_sample[-self.target_length:]

        if self.type == 0:
            return (sp_item, item_condition), sp_target
        elif self.type == 1:
            return (ap_item, item_condition), ap_target
        else:
            return (vuv_item, item_condition), vuv_target


    def __len__(self):

        return self._length