import torch
import torch.optim as optim
import torch.utils.data
import time
import os
from datetime import datetime
from torch.autograd import Variable
from model.util import *
import pyworld as pw
import matplotlib.pyplot as plt



class TimbreTrainer:
    def __init__(self,
                 model,
                 dataset,
                 device,
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 lr=0.0005,
                 weight_decay=0):

        self.model = model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.clip = None

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr / (1 + 0.00001 * epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, batch_size=32, epochs=10):
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=False)
        step = 0
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            self.adjust_learning_rate(current_epoch)
            tic = time.time()
            for (x, target) in iter(self.dataloader):
                x, condi = x
                x = x.to(self.device)
                condi = condi.to(self.device)

                target = target.to(self.device)

                output = self.model(x, condi)
                loss = CGM_loss(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.item()
                print('loss: ', loss)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                self.optimizer.step()
                step += 1

                # time step duration:
                if step == 100:
                    toc = time.time()
                    print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

                if step % self.snapshot_interval == 0:
                    self.save_model(current_epoch)

                    # gen = self.generate_audio()
                    # sp = pw.decode_spectral_envelope(gen.cpu().numpy().astype(np.double), 32000, 1024)
                    # plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom',interpolation='none')
                    # plt.show()

                #self.logger.log(step, loss)
        self.save_model(epochs)

    def validate(self):
        self.model.eval()


        return avg_loss, avg_accuracy

    def save_model(self, epoch):
        if self.snapshot_path is None:
            return
        time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot_path)
        torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + str(epoch) + '_' + time_string)
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