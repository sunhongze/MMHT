import math
import torch
import torch.nn as nn
from collections import OrderedDict
from .config import config
import torch.nn.functional as F

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = (config.alpha / 2) / (1 + ((config.alpha * math.pi / 2) * (x)).square()) * grad_output
        return grad_x

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):
        B, T = input.shape[0:2]
        output = self._conv_forward(input.flatten(0, 1).contiguous(), self.weight)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__(
            num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        output = []
        for t in range(input.shape[1]):
            output.append(F.batch_norm(input[:,t,...],
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps))
        return torch.stack(output, dim=1)

class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size):
        super(AdaptiveMaxPool2d, self).__init__(output_size)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.adaptive_max_pool2d(input.flatten(0, 1).contiguous(), self.output_size, self.return_indices)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__(output_size)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.adaptive_avg_pool2d(input.flatten(0, 1).contiguous(), self.output_size)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class LIF(nn.Module):
    def __init__(self, train=False, thresh=config.thresh, tau=config.tau):
        super(LIF, self).__init__()
        self.act = ZIF.apply
        self.tau = tau
        self.thresh = nn.Parameter(torch.ones(thresh) * config.thresh, requires_grad=train)

    def forward(self, input):
        T = input.shape[1]
        mem = 0
        spike_pot = []
        for t in range(T):
            mem = mem * self.tau + input[:, t, ...]
            spike = self.act(mem - self.thresh, config.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)

class AlexSNN(nn.Module):
    def __init__(self, output_layers):
        super(AlexSNN, self).__init__()
        self.output_layers = output_layers

        self.conv0 = Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=5)
        self.bn0   = BatchNorm2d(num_features=64)
        self.lif0  = LIF(train=True, thresh=[64, 72, 72])

        self.conv1 = Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn1   = BatchNorm2d(num_features=128)
        self.lif1  = LIF(train=True, thresh=[128, 36, 36])

        self.conv2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2   = BatchNorm2d(num_features=128)
        self.lif2  = LIF(train=True, thresh=[128, 36, 36])

        self.conv3 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3   = BatchNorm2d(num_features=256)
        self.lif3  = LIF(train=True, thresh=[256, 18, 18])

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, output_layers=None, ):
        outputs = OrderedDict()
        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv0(input.cuda())
        x = self.bn0(x)
        x = self.lif0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lif3(x)
        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x.mean(1)
        return name == output_layers[-1]

def alexsnn(output_layers=None, pretrained=False,):
    model = AlexSNN(output_layers)
    print('SNN backbone is Ture')
    return model
