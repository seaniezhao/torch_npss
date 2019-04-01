import os
import os.path
import time
from data.dataset import *
from model.util import *
import torch.nn as nn

class WaveNetModel(nn.Module):

    def __init__(self, hparams, device):

        super(WaveNetModel, self).__init__()

        self.type = hparams.type
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

        self.dilated_convs = nn.ModuleList()


        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.condi_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channel,
                                    out_channels=self.residual_channels,
                                    kernel_size=self.initial_kernel,
                                    bias=self.bias)


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




                # dilated convolutions
                self.dilated_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                    out_channels=self.dilation_channels*2,
                                                    kernel_size=self.kernel_size,
                                                    bias=self.bias,
                                                    dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=1,
                                                     bias=self.bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=1,
                                                 bias=self.bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv = nn.Conv1d(in_channels=self.skip_channels,
                                  out_channels=self.output_channel,
                                  kernel_size=1,
                                  bias=self.bias)

        # condition end conv
        self.cond_end_conv = nn.Conv1d(in_channels=self.condition_channel,
                                       out_channels=self.skip_channels,
                                       kernel_size=1,
                                       bias=self.bias)


        self.receptive_field = receptive_field + self.initial_kernel - 1

    def wavenet(self, input, condition):

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

            dilated = self.dilated_convs[i](x)
            # here plus condition

            condi = self.condi_convs[i](condition)
            condi = condi.expand(dilated.shape)
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

        if self.type == 2:
            x = torch.sigmoid(x)

        return x

    def forward(self, input, condition):
        x = self.wavenet(input, condition)

        return x

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def generate(self, conditions, cat_input=None, init_input=None):
        self.eval()

        if cat_input is not None:
            cat_input = cat_input.to(self.device)
            cat_input = torch.cat((torch.zeros(cat_input.shape[0], self.receptive_field).to(self.device), cat_input), 1)

        conditions = conditions.to(self.device)
        num_samples = conditions.shape[1]
        generated = torch.zeros(self.sample_channel, num_samples).cuda()

        skip_first20 = True

        if init_input is None:
            init_input = torch.zeros(self.sample_channel, self.receptive_field).to(self.device)
            skip_first20 = False
            if cat_input is not None:
                to_cat = cat_input[:, :self.receptive_field]
        else:
            init_input = init_input[:, :self.receptive_field].to(self.device)
            generated[:, :self.receptive_field] = init_input
            if cat_input is not None:
                to_cat = cat_input[:, self.receptive_field:self.receptive_field*2]

        model_input = init_input.unsqueeze(0)
        if cat_input is not None:
            model_input = torch.cat((init_input, to_cat), 0).unsqueeze(0)

        tic = time.time()
        for i in range(num_samples):
            if skip_first20 and i < self.receptive_field:
                continue
            condi = conditions[:, i].unsqueeze(0).unsqueeze(2)
            #  x shape : b, 240, l
            x = self.wavenet(model_input, condi).squeeze()
            x_sample = sample_from_CGM(x.detach())
            generated[:, i] = x_sample.squeeze(0)

            # set new input
            if i >= self.receptive_field:
                model_input = generated[:, i-self.receptive_field:i]
            else:
                # padding
                model_input = generated[:, 0:i+1]
                to_pad = init_input[:, i+1:]
                model_input = torch.cat((to_pad, model_input), 1)

            if cat_input is not None:
                model_input = torch.cat((model_input, cat_input[:, i:i+self.receptive_field]), 0)
            model_input = model_input.unsqueeze(0)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

        self.train()

        return generated

