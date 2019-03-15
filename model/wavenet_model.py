import os
import os.path
import time
from model.wavenet_modules import *
from data.dataset import *
from model.util import *


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=3,
                 blocks=2,
                 dilation_channels=130,
                 residual_channels=130,
                 skip_channels=240,
                 input_channel=60,
                 condition_channel=60,
                 cgm_factor=4,
                 output_length=32,
                 initial_kernel=10,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_channel = input_channel
        self.initial_kernel = initial_kernel
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.gcm_factor = cgm_factor
        self.condition_channel = condition_channel

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channel,
                                    out_channels=residual_channels,
                                    kernel_size=10,
                                    bias=bias)
        # condition start conv
        self.cond_start_conv = nn.Conv1d(in_channels=self.condition_channel,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)


        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            actual_layer = layers
            if b == blocks-1:
                actual_layer = layers - 1
            for i in range(actual_layer):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

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
                                  bias=True)

        # condition end conv
        self.cond_end_conv = nn.Conv1d(in_channels=self.condition_channel,
                                         out_channels=skip_channels,
                                         kernel_size=1,
                                         bias=bias)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field + self.initial_kernel - 1

    def wavenet(self, input, condition, dilation_func):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers - 1):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)

            # here plus condition
            #condi = self.cond_start_conv(condition)
            #residual = residual + condi

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                 s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        # plus condition
        #condi = self.cond_end_conv(condition)
        #skip = skip + condi

        x = torch.tanh(skip)
        x = self.end_conv(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input, condition):
        x = self.wavenet(input, condition,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]

        return x

    def generate_fast(self, num_samples):
        self.eval()


        # reset queues
        for queue in self.dilated_queues:
            queue.reset()



        input = torch.zeros(1, self.input_channel, self.initial_kernel)


        # generate new samples
        generated = torch.zeros(num_samples, self.input_channel)
        tic = time.time()
        for i in range(num_samples):
            #  x shape : b, 240, l
            x = self.wavenet(input, None,
                             dilation_func=self.queue_dilate).squeeze()

            x_sample = sample_from_CGM(x.detach())
            generated[i, :] = x_sample.squeeze(0)

            # set new input
            if i >= 10:
                input = generated[i-10:i, :]
            else:
            # padding
                input = generated[0:i+1, :]
                pad_dim = self.initial_kernel - (i + 1)
                input = torch.cat((torch.zeros(pad_dim, self.input_channel), input))


            input = input.unsqueeze(0).permute(0, 2, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

        self.train()
        return generated


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
        for q in self.dilated_queues:
            q.dtype = self.dtype
        super().cpu()



