import os
import os.path
import time
from model.wavenet_modules import *
from data.dataset import *
from model.util import *


class WaveNetModel(nn.Module):

    def __init__(self,
                 layers=3,
                 blocks=2,
                 dilation_channels=130,
                 residual_channels=130,
                 skip_channels=240,
                 input_channel=60,
                 condition_channel=1085,
                 cgm_factor=4,
                 initial_kernel=10,
                 kernel_size=2,
                 bias=True):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_channel = input_channel
        self.initial_kernel = initial_kernel
        self.kernel_size = kernel_size
        self.gcm_factor = cgm_factor
        self.condition_channel = condition_channel

        # build model
        receptive_field = 1
        init_dilation = 1
        #
        # self.dilations = []
        # self.dilated_queues = []

        self.dilated_convs = nn.ModuleList()


        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channel,
                                    out_channels=residual_channels,
                                    kernel_size=10,
                                    bias=bias)
        # condition start conv
        self.cond_start_conv = nn.Conv1d(in_channels=self.condition_channel,
                                    out_channels=dilation_channels*2,
                                    kernel_size=1,
                                    bias=bias)


        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            actual_layer = layers
            if b == blocks-1:
                actual_layer = layers - 1
            for i in range(actual_layer):

                # dilated convolutions
                self.dilated_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                    out_channels=dilation_channels*2,
                                                    kernel_size=kernel_size,
                                                    bias=bias,
                                                    dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=input_channel * cgm_factor,
                                  kernel_size=1,
                                  bias=bias)

        # condition end conv
        self.cond_end_conv = nn.Conv1d(in_channels=self.condition_channel,
                                       out_channels=skip_channels,
                                       kernel_size=1,
                                       bias=bias)


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
            condi = self.cond_start_conv(condition)
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

        return x


    def forward(self, input, condition):
        x = self.wavenet(input, condition)

        return x

    def generate(self, conditions, first_input=None):
        self.eval()
        conditions = conditions.cuda()

        num_samples = conditions.shape[1]

        if first_input is None:
            first_input = torch.zeros(self.input_channel, self.receptive_field).cuda()

        first_input = first_input[:, :self.receptive_field].cuda()
        model_input = first_input.unsqueeze(0)

        # generate new samples
        generated = torch.zeros(num_samples, self.input_channel).cuda()
        generated[:self.receptive_field, :] = first_input.transpose(0, 1)
        tic = time.time()
        for i in range(num_samples-self.receptive_field):
            condi = conditions[:,i+self.receptive_field].unsqueeze(0).unsqueeze(2)
            #  x shape : b, 240, l
            x = self.wavenet(model_input, condi).squeeze()

            x_sample = sample_from_CGM(x.detach())
            generated[i+self.receptive_field, :] = x_sample.squeeze(0)

            # set new input
            if i >= self.receptive_field:
                model_input = generated[i-self.receptive_field:i, :]
            else:
            # padding
                model_input = generated[0:i+1, :]
                to_pad = first_input.transpose(0, 1)[i+1:, :]
                model_input = torch.cat((to_pad, model_input))

            model_input = model_input.unsqueeze(0).permute(0, 2, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

        self.train()
        return generated


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s



