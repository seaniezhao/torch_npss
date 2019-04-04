import torch
import torch.optim as optim
import torch.utils.data
import time
import os
import torch.nn as nn
from data.dataset import TimbreDataset
from datetime import datetime
from torch.autograd import Variable
from model.util import *
import pyworld as pw
import matplotlib.pyplot as plt



class ModelTrainer:
    def __init__(self,
                 model,
                 data_folder,
                 device,
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 lr=0.0005,
                 weight_decay=0,
                 temperature=0.01):

        self.model = model

        self.trainset = TimbreDataset(data_folder=data_folder, receptive_field=model.receptive_field, type=model.model_type, train=True)
        print('the dataset has ' + str(len(self.trainset)) + ' items')

        self.testset = TimbreDataset(data_folder=data_folder, receptive_field=model.receptive_field, type=model.model_type, train=False)

        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.temperature = temperature

        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.clip = None

        self.device_count = torch.cuda.device_count()

        self.start_epoch = 0
        self.epoch = 0

    def adjust_learning_rate(self):

        real_epoch = self.start_epoch + self.epoch
        lr = self.lr / (1 + 0.00001 * real_epoch)

        print('lr '+str(lr)+'  epoch  '+str(real_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, batch_size=32, epochs=10):

        self.model.train()
        if self.device_count > 1:
            self.model = nn.DataParallel(self.model)
            print('multiple device using :', self.device_count)

        self.dataloader = torch.utils.data.DataLoader(self.trainset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=False)

        self.testdataloader = torch.utils.data.DataLoader(self.testset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          num_workers=8,
                                                          pin_memory=False)

        step = 0
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            self.epoch = current_epoch

            self.adjust_learning_rate()
            tic = time.time()

            total_loss = 0
            epoch_step = 0
            for (x, target) in iter(self.dataloader):
                x, condi = x
                x = x.to(self.device)
                condi = condi.to(self.device)

                target = target.to(self.device)

                output = self.model(x, condi)
                if self.model.model_type == 2:
                    loss = torch.mean((output.squeeze()-target.squeeze())**2)
                else:
                    loss = CGM_loss(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.item()
                total_loss += loss
                epoch_step += 1
                #print('loss: ', loss)

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                self.optimizer.step()
                step += 1

                # time step duration:
                if step == 100:
                    toc = time.time()
                    print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            self.save_model()
            test_loss = self.validate()
            toc = time.time()
            print("one epoch does take approximately " + str((toc - tic)) + " seconds), average loss: " + str(total_loss/epoch_step)+"  test loss: "+str(test_loss))

        self.save_model()


    def load_checkpoint(self, filename):

        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return self.start_epoch


    def validate(self):

        self.model.eval()

        total_loss = 0
        epoch_step = 0
        for (x, target) in iter(self.testdataloader):
            x, condi = x
            x = x.to(self.device)
            condi = condi.to(self.device)

            target = target.to(self.device)

            output = self.model(x, condi)
            if self.model.model_type == 2:
                loss = torch.mean((output.squeeze() - target.squeeze()) ** 2)
            else:
                loss = CGM_loss(output, target)

            loss = loss.item()

            total_loss += loss
            epoch_step += 1

        self.model.train()
        avg_loss = total_loss/epoch_step
        return avg_loss

    def save_model(self):
        if self.snapshot_path is None:
            return
        time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot_path)
        to_save = self.model
        if self.device_count > 1:
            to_save = self.model.module

        str_epoch = str(self.start_epoch + self.epoch)
        filename = self.snapshot_path + '/' + self.snapshot_name + '_' + str_epoch + '_' + time_string
        state = {'epoch': self.epoch + 1, 'state_dict': to_save.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

        print('model saved')


    # test
    def get_first_input(self):
        wav_path = './data/prepared_data/sp.npy'

        code_sp = np.load(wav_path).astype(np.double)
        return torch.Tensor(code_sp)

    def get_condition(self):
        c_path = './data/prepared_data/condition.npy'
        conditon = np.load(c_path).astype(np.float)
        return torch.Tensor(conditon).transpose(0, 1)

    def generate_audio(self):

        first_input = self.get_first_input()
        condi = self.get_condition()
        gen = self.model.generate(condi, first_input.transpose(0, 1))

        x = torch.sum((first_input - gen.cpu())**2)
        print("MSE !!!!!!!!!!!!!!", x)
        return gen