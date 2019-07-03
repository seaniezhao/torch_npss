import os
import os.path
import time
from data.dataset import *
from model.util import *
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class WaveNetModel(nn.Module):

    def __init__(self, hparams, device):

        super(WaveNetModel, self).__init__()
        self.noise_lambda = 0.2
        self.model_type = hparams.type
        self.layers = hparams.layers
        self.blocks = hparams.blocks
        self.dilation_channels = hparams.dilation_channels
        self.residual_channels = hparams.residual_channels
        self.skip_channels = hparams.skip_channels
        self.input_channel = hparams.input_channel
        self.initial_kernel = hparams.initial_kernel
        self.kernel_size = hparams.kernel_size
        self.output_channel = hparams.output_channel
        #  if use CGM sample_channel * cgm_factor = output_channel
        self.sample_channel = hparams.sample_channel
        self.condition_channel = hparams.condition_channel
        self.bias = hparams.bias

        self.device = device
        # build model
        receptive_field = 1
        init_dilation = 1
        #
        # self.dilations = []
        # self.dilated_queues = []

        self.dilated_pads = nn.ModuleList()
        self.dilated_convs = nn.ModuleList()

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.condi_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_pad = nn.ConstantPad1d((9, 0), 0)
        self.start_conv = nn.Conv1d(in_channels=self.input_channel,
                                    out_channels=self.residual_channels,
                                    kernel_size=self.initial_kernel,
                                    bias=self.bias)
        nn.init.xavier_uniform_(
            self.start_conv.weight, gain=nn.init.calculate_gain('linear'))

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            actual_layer = self.layers
            if b == self.blocks-1:
                actual_layer = self.layers - 1
            for i in range(actual_layer):

                self.condi_convs.append(nn.Conv1d(in_channels=self.condition_channel,
                                        out_channels=self.dilation_channels*2,
                                        kernel_size=1,
                                        bias=self.bias))
                nn.init.xavier_uniform_(
                    self.condi_convs[i].weight, gain=nn.init.calculate_gain('linear'))


                self.dilated_pads.append(nn.ConstantPad1d((new_dilation, 0), 0))
                # dilated convolutions
                self.dilated_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                    out_channels=self.dilation_channels*2,
                                                    kernel_size=self.kernel_size,
                                                    bias=self.bias,
                                                    dilation=new_dilation))
                nn.init.xavier_uniform_(
                    self.dilated_convs[i].weight, gain=nn.init.calculate_gain('linear'))


                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=1,
                                                     bias=self.bias))
                nn.init.xavier_uniform_(
                    self.residual_convs[i].weight, gain=nn.init.calculate_gain('linear'))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=1,
                                                 bias=self.bias))
                nn.init.xavier_uniform_(
                    self.skip_convs[i].weight, gain=nn.init.calculate_gain('linear'))


                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv = nn.Conv1d(in_channels=self.skip_channels,
                                  out_channels=self.output_channel,
                                  kernel_size=1,
                                  bias=self.bias)
        nn.init.xavier_uniform_(
            self.end_conv.weight, gain=nn.init.calculate_gain('linear'))

        # condition end conv
        self.cond_end_conv = nn.Conv1d(in_channels=self.condition_channel,
                                       out_channels=self.skip_channels,
                                       kernel_size=1,
                                       bias=self.bias)
        nn.init.xavier_uniform_(
            self.cond_end_conv.weight, gain=nn.init.calculate_gain('linear'))


        self.receptive_field = receptive_field + self.initial_kernel - 1

    def wavenet(self, input, condition):
        # input shape: (B,N,L) N is channel
        # condition shape: (B,cN,L) cN is condition channel

        input = self.start_pad(input)
        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers - 1):

            # |----------------------------------------------------|     *residual*
            # |                                                    |
            # |                |---- tanh --------|                |
            # -> dilate_conv ->|                  * ----|-- 1x1 -- + -->	*input*
            #                  |---- sigm --------|     |
            #                                          1x1
            #                                           |
            # ----------------------------------------> + ------------->	*skip*

            residual = x

            x = self.dilated_pads[i](x)
            dilated = self.dilated_convs[i](x)
            # here plus condition

            condi = self.condi_convs[i](condition)
            dilated = dilated + condi

            filter, gate = torch.chunk(dilated, 2, dim=1)

            # dilated convolution
            filter = torch.tanh(filter)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, -x.size(2):]

        # plus condition
        condi = self.cond_end_conv(condition)
        skip = skip + condi

        x = torch.tanh(skip)
        x = self.end_conv(x)

        if self.model_type == 2:
            x = torch.sigmoid(x)

        return x

    def forward(self, input, condition):
        # input noise
        # input shape : (B, N, L) for harmonic N = 60 = self.sample_channel
        # if self.sample_channel == 1:
        #     dist = Normal(input, self.noise_lambda)
        #     x = self.wavenet(dist.sample(), condition)
        # else:
        #     input = input.permute(0, 2, 1)
        #     sigmas = self.noise_lambda * torch.eye(self.sample_channel)
        #     sigmas = sigmas.repeat(input.shape[0], input.shape[1], 1, 1).to(self.device)
        #     dist = MultivariateNormal(input, sigmas)
        #     r_input = dist.sample().permute(0, 2, 1)
        #     x = self.wavenet(r_input, condition)

        dist = Normal(input, self.noise_lambda)
        x = self.wavenet(dist.sample(), condition)
        return x

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def generate(self, conditions, cat_input=None):
        # conditions shape: (condition_channel, len)
        self.eval()

        if cat_input is not None:
            cat_input = cat_input.to(self.device)

        conditions = conditions.to(self.device)
        num_samples = conditions.shape[1]
        generated = torch.zeros(self.sample_channel, num_samples).to(self.device)

        model_input = torch.zeros(1, self.input_channel, 1).to(self.device)

        from tqdm import tqdm
        for i in tqdm(range(num_samples)):
            if i < self.receptive_field:
                condi = conditions[:, :i + 1]
            else:
                condi = conditions[:, i - self.receptive_field + 1:i + 1]
            condi = condi.unsqueeze(0)

            x = self.wavenet(model_input, condi)
            x = x[:, :, -1].squeeze()
            if self.model_type == 2:
                x_sample = 0
                if x > 0.5:
                    x_sample = 1
                x_sample = torch.Tensor([x_sample]).to(self.device).unsqueeze(0)
            else:
                t = 0.01
                if self.model_type == 0:
                    t = 0.05
                x_sample = sample_from_CGM(x.detach(), t)

            generated[:, i] = x_sample.squeeze(0)

            # set new input
            if i < self.receptive_field - 1:
                model_input = generated[:, :i + 1]
                if cat_input is not None:
                    to_cat = cat_input[:, :i + 1]
                    model_input = torch.cat((to_cat, model_input), 0)

                model_input = torch.Tensor(np.pad(model_input.cpu(), ((0, 0), (1, 0)), 'constant', constant_values=0)).to(
                    self.device)
            else:
                model_input = generated[:, i - self.receptive_field + 1:i + 1]
                if cat_input is not None:
                    to_cat = cat_input[:, i - self.receptive_field + 1:i + 1]
                    model_input = torch.cat((to_cat, model_input), 0)

            model_input = model_input.unsqueeze(0)

        self.train()
        return generated

