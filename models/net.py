import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x







import torch
import torch.nn as nn
import math


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.stage1 = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.stage2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        input_channel = output_channel
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)),

        output_channel = 1280
        self.stage3 = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.linear = nn.Linear(output_channel, num_classes),


        self._initialize_weights()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.stage3(x)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np


def conv3x3(in_channels, out_channels, stride, padding=1, groups=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding,
                     groups=groups,
                     bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


class ShufflenetUnit(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShufflenetUnit, self).__init__()
        self.downsample = downsample

        if not self.downsample:  # ---if not downsample, then channel split, so the channel become half
            inplanes = inplanes // 2
            planes = planes // 2

        self.conv1x1_1 = conv1x1(in_channels=inplanes, out_channels=planes)
        self.conv1x1_1_bn = nn.BatchNorm2d(planes)

        self.dwconv3x3 = conv3x3(in_channels=planes, out_channels=planes, stride=stride, groups=planes)
        self.dwconv3x3_bn = nn.BatchNorm2d(planes)

        self.conv1x1_2 = conv1x1(in_channels=planes, out_channels=planes)
        self.conv1x1_2_bn = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def _channel_split(self, features, ratio=0.5):
        """
        ratio: c'/c, default value is 0.5
        """
        size = features.size()[1]
        split_idx = int(size * ratio)
        return features[:, :split_idx], features[:, split_idx:]

    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1]
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1).type(torch.long).cuda()
        features = features[:, index]
        return features

    def forward(self, x):
        if self.downsample:
            # x1 = x.clone() #----deep copy x, so where x2 is modified, x1 not be affected
            x1 = x
            x2 = x
        else:
            x1, x2 = self._channel_split(x)

        # ----right branch-----
        x2 = self.conv1x1_1(x2)
        x2 = self.conv1x1_1_bn(x2)
        x2 = self.relu(x2)

        x2 = self.dwconv3x3(x2)
        x2 = self.dwconv3x3_bn(x2)

        x2 = self.conv1x1_2(x2)
        x2 = self.conv1x1_2_bn(x2)
        x2 = self.relu(x2)

        # ---left branch-------
        if self.downsample:
            x1 = self.downsample(x1)

        x = torch.cat([x1, x2], 1)
        x = self._channel_shuffle(x)
        return x


class ShuffleNet(nn.Module):
    def __init__(self, feature_dim, layers_num, num_classes=1000):
        super(ShuffleNet, self).__init__()
        dim1, dim2, dim3, dim4, dim5 = feature_dim
        self.conv1 = conv3x3(in_channels=3, out_channels=dim1,
                             stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_layer(dim1, dim2, layers_num[0])
        self.stage2 = self._make_layer(dim2, dim3, layers_num[1])
        self.stage3 = self._make_layer(dim3, dim4, layers_num[2])

        self.conv5 = conv1x1(in_channels=dim4, out_channels=dim5)
        self.globalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(dim5, num_classes)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """

    def _make_layer(self, dim1, dim2, blocks_num):
        half_channel = dim2 // 2
        downsample = nn.Sequential(
            conv3x3(in_channels=dim1, out_channels=dim1, stride=2, padding=1, groups=dim1),
            nn.BatchNorm2d(dim1),
            conv1x1(in_channels=dim1, out_channels=half_channel),
            nn.BatchNorm2d(half_channel),
            nn.ReLU(inplace=True)
        )

        layers = []
        layers.append(ShufflenetUnit(dim1, half_channel, stride=2, downsample=downsample))
        for i in range(blocks_num):
            layers.append(ShufflenetUnit(dim2, dim2, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("x0.size:\t", x.size())
        x = self.maxpool(x)
        # print("x1.size:\t", x.size())
        x = self.stage1(x)
        # print("x2.size:\t", x.size())
        x = self.stage2(x)
        # print("x3.size:\t", x.size())
        x = self.stage3(x)
        # print("x4.size:\t", x.size())

        x = self.conv5(x)
        # print("x5.size:\t", x.size())
        x = self.globalpool(x)
        # print("x6.size:\t", x.size())

        x = x.view(-1, 1024)
        x = self.fc(x)

        return x


features = {
    "0.5x": [24, 48, 96, 192, 1024],
    "1x": [24, 116, 232, 464, 1024],
    "1.5x": [24, 176, 352, 704, 1024],
    "2x": [24, 244, 488, 976, 2048]
}


def shufflenet():
    model = ShuffleNet(features["0.5x"], [3, 7, 3])
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all___ = [
    'PPLCNet_x0_25', 'PPLCNet_x0_35', 'PPLCNet_x0_5', 'PPLCNet_x0_75',
    'PPLCNet_x1_0', 'PPLCNet_x1_5', 'PPLCNet_x2_0', 'PPLCNet_x2_5'
]


def swish(x):
    return x * x.sigmoid()


def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0, ),
            nn.ReLU(),
            nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
            HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]

        padding = (kernel_size - 1) // 2

        if use_se:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),

                # SE
                SELayer(inp, inp),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish(),

            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish()
            )

    def forward(self, x):
        return self.conv(x)


class PPLCNet(nn.Module):
    def __init__(self, scale=1.0, num_classes=1000, dropout_prob=0.2):
        super(PPLCNet, self).__init__()
        self.cfgs = [
            # k,  c,  s, SE
            [3, 32, 1, 0],

            [3, 64, 2, 0],
            [3, 64, 1, 0],

            [3, 128, 2, 0],
            [3, 128, 1, 0],

            [5, 256, 2, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],

            [5, 512, 2, 1],
            [5, 512, 1, 1],
        ]
        self.scale = scale

        input_channel = _make_divisible(16 * scale)
        layer1 = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), HardSwish()]
        layer2 = []
        layer3 = []
        block = DepSepConv
        num = 0
        for k, c, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * scale)
            if num < 5:
                layer1.append(block(input_channel, output_channel, k, s, use_se))
            elif num < 11:
                layer2.append((block(input_channel, output_channel, k, s, use_se)))
            else:
                layer3.append((block(input_channel, output_channel, k, s, use_se)))
            input_channel = output_channel
            num = num + 1

        # self.features = nn.Sequential(*layers)
        self.stage1 = nn.Sequential(*layer1)
        self.stage2 = nn.Sequential(*layer2)
        self.stage3 = nn.Sequential(*layer3)

        # # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(input_channel, 1280, 1, 1, 0)
        self.hwish = HardSwish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.hwish(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def PPLCNet_x0_25(**kwargs):
    """
    Constructs PPLCNet_x0_25 model
    """
    model = PPLCNet(scale=0.25, **kwargs)

    return model


def PPLCNet_x0_35(**kwargs):
    """
    Constructs PPLCNet_x0_35 model
    """
    model = PPLCNet(scale=0.35, **kwargs)

    return model


def PPLCNet_x0_5(**kwargs):
    """
    Constructs PPLCNet_x0_5 model
    """
    model = PPLCNet(scale=0.5, **kwargs)

    return model


def PPLCNet_x0_75(**kwargs):
    """
    Constructs PPLCNet_x0_75 model
    """
    model = PPLCNet(scale=0.75, **kwargs)

    return model


def PPLCNet_x1_0(**kwargs):
    """
    Constructs PPLCNet_x1_0 model
    """
    model = PPLCNet(scale=1.0, **kwargs)

    return model


def PPLCNet_x1_5(**kwargs):
    """
    Constructs PPLCNet_x1_5 model
    """
    model = PPLCNet(scale=1.5, **kwargs)

    return model


def PPLCNet_x2_0(**kwargs):
    """
    Constructs PPLCNet_x2_0 model
    """
    model = PPLCNet(scale=2.0, **kwargs)

    return model


def PPLCNet_x2_5(**kwargs):
    """
    Constructs PPLCNet_x2_5 model
    """
    model = PPLCNet(scale=2.5, **kwargs)

    return model
