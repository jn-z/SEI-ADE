import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
import math
import functools
import numpy as np
from collections import OrderedDict
from torch.nn import Parameter
import pdb
from collections import OrderedDict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, layers=[3, 4, 6, 3], flatten_dim=4096, spkVec_dim=256):
        self.feature_maps = [16, 16, 32, 64, 128]
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, self.feature_maps[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.feature_maps[0])
        self.relu1 = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.feature_maps[1], layers[0], stride=1)
        self.layer2 = self._make_layer(self.feature_maps[2], layers[1], stride=2)
        self.layer3 = self._make_layer(self.feature_maps[3], layers[2], stride=2)
        # self.layer4 = self._make_layer(self.feature_maps[4], layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(flatten_dim, spkVec_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(spkVec_dim)

        # pdb.set_trace()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion

        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        # x = torch.cat((x, x), 3)
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.maxpool(x)

        x_1 = self.layer1(x)

        x_2 = self.layer2(x_1)
        x_4 = self.layer3(x_2)
        # x_4 = self.layer4(x_3)

        mean_x_4 = torch.mean(x_4, dim=2)
        var_x_4 = torch.var(x_4, dim=2)
        # std_x_5=torch.std(x_5,dim=1,unbiased=False)
        x_5 = torch.cat((mean_x_4, torch.sqrt(var_x_4 + 0.00001)), dim=2)
        x_5 = x_5.contiguous().view(x.size(0), -1)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # import pdb
        x_6 = self.fc(x_5)
        x_6 = self.relu2(x_6)
        speakerVector = self.bn2(x_6)

        return speakerVector


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=1, output_nc=24, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        #        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class TDNN(nn.Module):
    def __init__(self, input_dim=64, output_dim=512, context_size=5, stride=1, dilation=1, batch_norm=True):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert d == self.input_dim
        x = x.unsqueeze(1)
        # Unfold input into smaller temporal contexts
        x = F.unfold(x, (self.context_size, self.input_dim), stride=(1, self.input_dim), dilation=(self.dilation, 1))
        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        if self.batch_norm:
            x = x.transpose(1, 2).contiguous()
            x = self.bn(x)
            x = x.transpose(1, 2).contiguous()

        return x


class CNN(nn.Module):
    def __init__(self, kernel_size=5, pool_size=2, stride=1, padding=2):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.feature_maps = [64, 128, 256, 512, 512]

        self.layer6 = nn.Sequential()
        self.layer6.add_module('linear6', nn.Linear(3072, 512))
        self.layer6.add_module('tdnn6_relu', torch.nn.ReLU())
        self.layer6.add_module('tdnn6_bn', nn.BatchNorm1d(512))
        self.layer7 = nn.Sequential()
        # self.layer7.add_module('dropout7', nn.Dropout(p=0.2))
        self.layer7.add_module('linear7', nn.Linear(512, 512))
        self.layer7.add_module('tdnn7_relu', torch.nn.ReLU())
        self.layer7.add_module('tdnn7_bn', nn.BatchNorm1d(512))

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x_0):
        # pdb.set_trace()
        # x_1 = self.layer1(x_0)
        # x_2 = self.layer2(x_1)
        # x_3 = self.layer3(x_2)
        # x_4 = self.layer4(x_3)
        # x_5 = self.layer5(x_4)
        mean_x_5 = torch.mean(x_0, dim=1)
        var_x_5 = torch.var(x_0, dim=1)
        # std_x_5=torch.std(x_5,dim=1,unbiased=False)
        statistic_x_5 = torch.cat((mean_x_5, torch.sqrt(var_x_5 + 0.00001)), dim=1)
        x_6 = self.layer6(statistic_x_5)
        speakerVector = self.layer7(x_6)
        return speakerVector


class CNNnet(nn.Module):
    def __init__(self, flatten_dim=4096, spkVec_dim=256):
        super(CNNnet, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module('conv1', BN_Conv2d(1, 16, 3, 2, 1))
        self.layer.add_module('relu1', nn.ReLU())

        self.layer.add_module('conv2', BN_Conv2d(16, 32, 3, 2, 1))
        self.layer.add_module('relu2', nn.ReLU())

        self.layer.add_module('conv3', BN_Conv2d(32, 64, 3, 2, 1))
        self.layer.add_module('relu3', nn.ReLU())

        self.layer.add_module('conv4', BN_Conv2d(64, 64, 2, 2, 0))
        self.layer.add_module('relu4', nn.ReLU())

        self.fl = nn.Sequential()
        self.fl.add_module('Linear_last', nn.Linear(flatten_dim, spkVec_dim))
        self.fl.add_module('ReLU_last', nn.ReLU(inplace=True))
        self.fl.add_module('bn_last', nn.BatchNorm1d(spkVec_dim))

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        x = torch.cat((x, x), 3)
        # pdb.set_trace()
        x = self.layer(x)
        out = x.view(x.size(0), -1)

        # pdb.set_trace()
        out = self.fl(out)

        # out:64*256
        return out


class RNNnet(nn.Module):
    def __init__(self, flatten_dim=4096, spkVec_dim=256):
        super(RNNnet, self).__init__()
        self.hidden_size = flatten_dim
        self.rnn = nn.RNN(
            input_size=40 * spkVec_dim,
            hidden_size=flatten_dim,
            num_layers=1,
            batch_first=True
        )

        self.fl = nn.Sequential()
        self.fl.add_module('Linear_last', nn.Linear(flatten_dim, spkVec_dim))
        self.fl.add_module('ReLU_last', nn.ReLU(inplace=True))
        self.fl.add_module('bn_last', nn.BatchNorm1d(spkVec_dim))

    def initial_parameters(self):
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        # pdb.set_trace()
        x = x.contiguous().view(x.size(0), -1, x.size(1) * x.size(2))
        # pdb.set_trace()
        out, ht = self.rnn(x)
        # pdb.set_trace()
        out = out.view(-1, self.hidden_size)
        out = self.fl(out)
        # pdb.set_trace()
        # out = out.unsqueeze(dim=0)
        return out


class DNNnet(nn.Module):
    def __init__(self, flatten_dim=4096, spkVec_dim=256):
        super(DNNnet, self).__init__()
        self.hidden_size = flatten_dim
        self.dnn = nn.Sequential()
        self.dnn.add_module('linear1', nn.Linear(40 * spkVec_dim, flatten_dim + 128))
        # self.dnn.add_module('dropout1', nn.Dropout(0.5))
        self.dnn.add_module('ELU1', nn.ELU())

        self.dnn.add_module('linear2', nn.Linear(flatten_dim + 128, flatten_dim))
        # self.dnn.add_module('dropout2', nn.Dropout(0.5))
        self.dnn.add_module('ELU2', nn.ELU())

        self.dnn.add_module('linear3', nn.Linear(flatten_dim, spkVec_dim))
        # self.dnn.add_module('dropout3', nn.Dropout(0.5))
        self.dnn.add_module('ELU3', nn.ELU())

        self.fl = nn.Sequential()
        self.fl.add_module('Linear_last', nn.Linear(128, spkVec_dim))
        self.fl.add_module('ReLU_last', nn.ReLU(inplace=True))
        self.fl.add_module('bn_last', nn.BatchNorm1d(spkVec_dim))

    def initial_parameters(self):
        for p in self.dnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1) * x.size(2))
        # pdb.set_trace()
        out = self.dnn(x)
        # out = x.view(x.size(0), -1)

        # pdb.set_trace()
        # out = self.fl(out)

        out = out.view(out.size(0), -1)
        # pdb.set_trace()
        # out:64*256
        return out


class Lstm(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, spkVec_dim=256, num_layer=2):
        super(Lstm, self).__init__()
        self.layer1 = nn.LSTM(spkVec_dim, hidden_size, num_layer)
        self.layer2 = nn.Linear(40 * hidden_size, spkVec_dim)

    def forward(self, x):
        # x = x.contiguous().view(x.size(0),-1, x.size(1), x.size(2))

        x, _ = self.layer1(x)
        # pdb.set_trace()
        s, b, h = x.size()
        x = x.view(s, -1)
        # pdb.set_trace()
        x = self.layer2(x)
        # x = x.view(s, b, -1)

        # pdb.set_trace()
        return x


class fullyConnect(nn.Module):
    def __init__(self, target_num=10000, spkVec_dim=256):
        super(fullyConnect, self).__init__()
        self.spkVec_dim = spkVec_dim
        self.target_num = target_num
        self.layer1 = nn.Sequential()
        self.layer1.add_module('linear1', nn.Linear(self.spkVec_dim, self.target_num))
        self.initial_parameters()

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        # pdb.set_trace()
        hiddenVec = self.layer1(x)
        tar = F.softmax(hiddenVec, dim=1)
        return tar


class fullyConnectLMCL(nn.Module):
    def __init__(self, target_num=10000, spkVec_dim=512):
        super(fullyConnectLMCL, self).__init__()
        self.target_num = target_num
        self.spkVec_dim = spkVec_dim
        self.pad_cols = nn.Parameter(torch.ones(1, 1), requires_grad=False)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('linear1', nn.Linear(self.spkVec_dim, self.target_num))

        self.initial_parameters()

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, speakerVector, lmcl_margin):
        # hiddenVec = self.layer2(speakerVector)

        dropedSpeakerVector = F.dropout(speakerVector, p=0.4, training=self.training)

        lastWeight = self.layer1.linear1.weight
        lastBias = self.layer1.linear1.bias.unsqueeze(0)

        pad_cols = self.pad_cols.repeat(dropedSpeakerVector.size(0), 1)
        # test_index_margin = torch.nonzero(lmcl_margin)

        cattedWeight = torch.cat((lastWeight.t(), lastBias), 0)
        cattedSpeakerVector = torch.cat((dropedSpeakerVector, pad_cols), 1)
        # debug
        '''
        hidden_test = torch.matmul(cattedSpeakerVector, cattedWeight)
        hidden_test = torch.matmul(dropedSpeakerVector, lastWeight.t()) + lastBias.expand(dropedSpeakerVector.size(0), lastBias.size(1))
        delta_hidden = hidden_test - hiddenVec
        '''
        # L2 norm in the SpeakerVector dimension
        catted_weight_length = torch.norm(cattedWeight, 2, 0)
        catted_speakerVector_length = torch.norm(cattedSpeakerVector, 2, 1)

        catted_weight_norm = catted_weight_length.contiguous().view(1, -1).expand(cattedWeight.size())
        catted_speakerVector_norm1 = catted_speakerVector_length.contiguous().view(-1, 1).expand(
            cattedSpeakerVector.size())
        catted_speakerVector_norm2 = catted_speakerVector_length.contiguous().view(-1, 1).expand(
            cattedSpeakerVector.size(0), cattedWeight.size(1))

        normedWeight = cattedWeight / catted_weight_norm
        normedSpeakerVector = cattedSpeakerVector / catted_speakerVector_norm1
        pscore = catted_speakerVector_norm2 * (torch.matmul(normedSpeakerVector, normedWeight) - lmcl_margin)

        tar = F.softmax(pscore, dim=1)
        return speakerVector, tar


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, target_num=10000, spkVec_dim=512, s=64.0, m=0.25, is_anneal=False, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        # pdb.set_trace()
        self.in_features = spkVec_dim
        self.out_features = target_num
        self.s = s
        self.m = m
        self.is_anneal = is_anneal
        if self.is_anneal:
            self.lambda_base = 1000
            self.lambda_gamma = 0.001
            self.lambda_power = 3
            self.lambda_min = 0
            self.iter = 0

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.layer1 = nn.Sequential()
        self.layer1.add_module('linear1', nn.Linear(spkVec_dim, target_num))
        self.initial_parameters()

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        weight = self.layer1.linear1.weight
        # pdb.set_trace()
        # cos(theta)
        cosine = F.linear(F.normalize(input), F.normalize(weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        if self.is_anneal:
            self.iter += 1
            self.lamb = max(self.lambda_min,
                            self.lambda_base * (1 + self.lambda_gamma * self.iter) ** (-1 * self.lambda_power))
            phi = (self.lamb * cosine + phi) / (1 + self.lamb)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        tar = F.softmax(output, dim=1)
        # NormOfFeature = torch.norm(input, 2, 1)
        # output *= NormOfFeature.view(-1, 1)
        return tar, weight


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, target_num=10000, spkVec_dim=512, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = spkVec_dim
        self.out_features = target_num
        self.s = s
        self.m = m
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('linear1', nn.Linear(spkVec_dim, target_num))
        self.initial_parameters()

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # pdb.set_trace()
        weight = self.layer1.linear1.weight
        cosine = F.linear(F.normalize(input), F.normalize(weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        tar = F.softmax(output, dim=1)
        return tar, weight


class SphereMarginProduct(nn.Module):
    def __init__(self, target_num=10000, spkVec_dim=512, m=2):
        super(SphereMarginProduct, self).__init__()
        # pdb.set_trace()
        # self.args = args
        self.in_features = spkVec_dim
        self.out_features = target_num
        self.m = int(m)
        self.base = 1000.0
        if self.m == 2:
            self.gamma = 0.12
        else:
            self.gamma = 0.01
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('linear1', nn.Linear(spkVec_dim, target_num))
        self.initial_parameters()

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, input, label):
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
        # pdb.set_trace()
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        weight = self.layer1.linear1.weight
        cos_theta = F.linear(F.normalize(input), F.normalize(weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)
        tar = F.softmax(output, dim=1)

        # tar_select_new = torch.gather(tar, 1, label)
        # ce_loss = -torch.log(tar_select_new+pow(10.0,-8))
        # if self.iter % self.args.display_freq== 0:
        #     if self.args.rank == 0:
        #         print('######ce_loss.sum() = %.5f'%ce_loss.sum())
        #         print('######self.lamb  = %.5f'% self.lamb )
        return tar, weight


class MHELossFun(nn.Module):
    def __init__(self, mhe_type=1, mhe_lambda=0.01):
        super(MHELossFun, self).__init__()
        self.mhe_lambda = mhe_lambda
        self.mhe_type = mhe_type

    def forward(self, labels, weight):
        trans_w = F.normalize(weight)  # [100,512]
        w_norm = trans_w.t()  # [512,100]
        num_classes = trans_w.shape[0]

        # np.savetxt("./weight.txt", weight.t().cpu().detach().numpy(),fmt='%f',delimiter=',')
        # np.savetxt("./label.txt", labels.squeeze(1).cpu().detach().numpy(),fmt='%f',delimiter=',')
        if self.mhe_type == 1:
            # pdb.set_trace()
            sel_w = torch.index_select(trans_w, 0, labels.squeeze(1))
            batch_size = sel_w.shape[0]
            # Version 1
            # The inner product may be faster than the element-wise operations.
            dist = (2.0 - 2.0 * torch.matmul(sel_w, w_norm))  # [10,100]
            mask = torch.zeros(batch_size, num_classes).cuda()
            mask.scatter_(1, labels.view(-1, 1), 1)

            dist = (1.0 - mask) * dist + mask * 1e6
            mhe_loss = self.mhe_lambda * (torch.sum((1.0 - mask) * (1.0 / dist)) /
                                          float(batch_size * (num_classes - 1)))

        elif self.mhe_type == 2:
            dist = 2.0 - 2.0 * torch.matmul(trans_w, w_norm)
            mask = torch.eye(num_classes)
            dist = (1.0 - mask) * dist + mask * 1e6
            mhe_loss = self.mhe_lambda * (torch.sum((1.0 - mask) * (1.0 / dist)) /
                                          float(batch_size * (num_classes - 1)))
        else:
            raise ValueError("Not implemented.")
        return mhe_loss


# alexnet
class AlexNet(nn.Module):
    def __init__(self, spkVec_dim=256):
        super().__init__()

        # 第一层是 5x5 的卷积，输入的channels 是 3，输出的channels是 64,步长 1,没有 padding
        # Conv2d 的第一个参数为输入通道，第二个参数为输出通道，第三个参数为卷积核大小
        # ReLU 的参数为inplace，True表示直接对输入进行修改，False表示创建新创建一个对象进行修改
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU()
        )

        # 第二层为 3x3 的池化，步长为2，没有padding
        self.max_pool1 = nn.MaxPool2d(3, 2)

        # 第三层是 5x5 的卷积， 输入的channels 是64，输出的channels 是64，没有padding
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1),
            nn.ReLU()
        )

        # 第四层是 3x3 的池化， 步长是 2，没有padding
        self.max_pool2 = nn.MaxPool2d(3, 2)

        out_hight = math.floor((math.floor((((64 - 5 + 1) - 3) / 2 + 1) - 5 + 1) - 3) / 2 + 1)
        out_wight = math.floor((math.floor((((256 - 5 + 1) - 3) / 2 + 1) - 5 + 1) - 3) / 2 + 1)

        # pdb.set_trace()
        # 第五层是全连接层，输入是 1204 ，输出是384  out_hight * out_wight * 16
        self.fc1 = nn.Sequential(
            nn.Linear(23808, 384),
            nn.ReLU()
        )

        # 第六层是全连接层，输入是 384， 输出是192
        self.fc2 = nn.Sequential(
            nn.Linear(384, spkVec_dim),
            nn.ReLU()
        )

        # # 第七层是全连接层，输入是192， 输出是 10
        # self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        # pdb.set_trace()
        # 将图片矩阵拉平
        x = x.view(x.shape[0], -1)

        # pdb.set_trace()
        x = self.fc1(x)
        x = self.fc2(x)

        # pdb.set_trace()

        # x = self.fc3(x)
        return x


# googlenet
class GoogLeNet(nn.Module):

    def __init__(self, spkVec_dim=256, aux_logits=True, transform_input=False, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # pdb.set_trace()
        self.inception3a = Inception_googlenet(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_googlenet(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception_googlenet(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_googlenet(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_googlenet(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_googlenet(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_googlenet(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception_googlenet(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_googlenet(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, spkVec_dim)
            self.aux2 = InceptionAux(528, spkVec_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, spkVec_dim)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        # pdb.set_trace()
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224   原始输入数据    大小为224*224 通道数为3
        x = self.conv1(x)  # 图1 字母A 对应部分   out size = N x 64 x 112 x 112
        # N x 64 x 112 x 112
        x = self.maxpool1(x)  # out size = N x 64 x 56 x 56
        # N x 64 x 56 x 56
        x = self.conv2(x)  # 图1 字母B 对应部分    out size = N x 64 x 56 x 56
        # N x 64 x 56 x 56
        x = self.conv3(x)  # 图1 字母C 对应部分    out size = N x 192 x 56 x 56
        # N x 192 x 56 x 56
        x = self.maxpool2(x)  # out size = N x 192 x 28 x 28

        # pdb.set_trace()

        # N x 192 x 28 x 28
        x = self.inception3a(x)  # 对应图2   其他的inception层与他的结构是一样的 不同的只是输入和输出
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)  # 对应图3 是一个输出口   这个是第一次输出 后面还有2个输出  结构一样 是更深层的输出罢了

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        # pdb.set_trace()
        # if self.training and self.aux_logits:
        # return _GoogLeNetOuputs(x, aux2, aux1)
        # pdb.set_trace()
        return x


class Inception_googlenet(nn.Module):  # 对应图 2   实现inception块的各种操作

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception_googlenet, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)  # Inception 层的 1*1卷积层

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),  # Inception 层的 1*1卷积  + 3*3卷积
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),  # Inception 层的 1*1卷积  + 5*5卷积
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(  # Inception 层的 max池化  + 1*1卷积其中MaxPool2d步长为1 不影响特征图的尺寸。
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):  # 下面注释只针对inception3a 这个的  其他的需要相应调整这看
        branch1 = self.branch1(x)  # branch1   N x 64 x 28 x 28
        branch2 = self.branch2(x)  # branch2   N x 128 x 28 x 28
        branch3 = self.branch3(x)  # branch3   N x 32 x 28 x 28
        branch4 = self.branch4(x)  # branch4   N x 32 x 28 x 28
        # 最终将四组结果拼接到一起
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # outputs  N x 256 x 28 x 28


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.view(x.size(0), -1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes

        return x


class BasicConv2d(nn.Module):  # Conv2d+BN+Relu

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# VGG系列
class VGG(nn.Module):
    """
    VGG builder
    """

    def __init__(self, arch: object, flatten_dim=4096, spkVec_dim=256) -> object:
        super(VGG, self).__init__()
        self.in_channels = 1
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(2 * flatten_dim, flatten_dim)
        self.bn1 = nn.BatchNorm1d(flatten_dim)
        self.bn2 = nn.BatchNorm1d(flatten_dim)
        self.fc2 = nn.Linear(flatten_dim, flatten_dim)
        self.fc3 = nn.Linear(flatten_dim, spkVec_dim)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.cat((x, x), 1)
        # pdb.set_trace()
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))

        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)

        # pdb.set_trace()
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.fc3(out)
        # pdb.set_trace()
        # return F.softmax(self.fc3(out))
        return out


def VGG_11():
    return VGG([1, 1, 2, 2, 2], num_classes=1000)


def VGG_13():
    return VGG([1, 1, 2, 2, 2], num_classes=1000)


def VGG_16():
    return VGG([2, 2, 3, 3, 3], num_classes=1000)


def VGG_19():
    return VGG([2, 2, 4, 4, 4], num_classes=1000)


# densenet系列
class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(in_channels + growth_rate * i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, spkVec_dim=10, channel=1):
        super(DenseNet, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        # 表示cifar-10
        if spkVec_dim == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channel, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channel, num_init_feature,
                                    kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i + 1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(num_feature, spkVec_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        # pdb.set_trace()
        out = self.features(x)
        # pdb.set_trace()

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # pdb.set_trace()
        return out


# DenseNet_BC for ImageNet
def DenseNet121():
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000)


def DenseNet169():
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=1000)


def DenseNet201():
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=1000)


def DenseNet161():
    return DenseNet_BC(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=1000)


# DenseNet_BC for cifar
def densenet_100():
    return DenseNet(growth_rate=12, block_config=(16, 16, 16), num_classes=1000)


# inception系列
class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))


class Inception_A(nn.Module):
    '''
    (in_size, 96, 96, 64, 96, 64, 96, 96)
    '''

    def __init__(self, in_size, b1, b2, b3_n1, b3_n2, b4_n1, b4_n2, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_size, b1, (1, 1), 1, 0, bias=False)
        )
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_size, b2, (1, 1), 1, 0, bias=False)
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_size, b3_n1, (1, 1), 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n2, 3, 1, 1, bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_size, b4_n1, (1, 1), 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n2, 3, 1, 1, bias=False),
            BN_Conv2d(b4_n2, b4_n3, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)
        return out


class Reduction_A(nn.Module):
    """
    Reduction-A block for Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2 nets
    """

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch2 = BN_Conv2d(in_channels, n, 3, 2, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, k, 1, 1, 0, bias=False),
            BN_Conv2d(k, l, 3, 1, 1, bias=False),
            BN_Conv2d(l, m, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Inception_A_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1\
    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n3, b3_n1, b3_n3_1, b3_n3_2, n1_linear):
        super(Inception_A_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 1, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3_1, 3, 1, 1, bias=False),
            BN_Conv2d(b3_n3_1, b3_n3_2, 3, 1, 1, bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n3 + b3_n3_2, n1_linear, 1, 1, 0, bias=True)

        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat((out1, out2, out3), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class Inception_B(nn.Module):
    """
    Inception-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x7, b3_n7x1, b4_n1, b4_n1x7_1,
                 b4_n7x1_1, b4_n1x7_2, b4_n7x1_2):
        super(Inception_B, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x7_1, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_1, b4_n7x1_1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b4_n7x1_1, b4_n1x7_2, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_2, b4_n7x1_2, (7, 1), (1, 1), (3, 0), bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Inception_B_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1\
    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x7, b2_n7x1, n1_linear):
        super(Inception_B_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b2_n1x7, b2_n7x1, (7, 1), (1, 1), (3, 0), bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n7x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class Reduction_B_v4(nn.Module):
    """
    Reduction-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False)
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Reduction_B_Res(nn.Module):
    """
    Reduction-B block for Inception-ResNet-v1 \
    and Inception-ResNet-v2  net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n3, b4_n1, b4_n3_1, b4_n3_2):
        super(Reduction_B_Res, self).__init__()
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3, 3, 2, 0, bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n3_1, 3, 1, 1, bias=False),
            BN_Conv2d(b4_n3_1, b4_n3_2, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Inception_C(nn.Module):
    """
    Inception-C block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x3_3x1, b4_n1,
                 b4_n1x3, b4_n3x1, b4_n1x3_3x1):
        super(Inception_C, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3_1 = BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False)
        self.branch3_1x3 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch3_3x1 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)
        self.branch4_1 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            BN_Conv2d(b4_n1x3, b4_n3x1, (3, 1), (1, 1), (1, 0), bias=False)
        )
        self.branch4_1x3 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch4_3x1 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        tmp = self.branch3_1(x)
        out3_1 = self.branch3_1x3(tmp)
        out3_2 = self.branch3_3x1(tmp)
        tmp = self.branch4_1(x)
        out4_1 = self.branch4_1x3(tmp)
        out4_2 = self.branch4_3x1(tmp)
        return torch.cat((out1, out2, out3_1, out3_2, out4_1, out4_2), 1)


class Inception_C_res(nn.Module):
    """
    Inception-C block for Inception-ResNet-v1\
    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x3, b2_n3x1, n1_linear):
        super(Inception_C_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            BN_Conv2d(b2_n1x3, b2_n3x1, (3, 1), (1, 1), (1, 0), bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n3x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class Stem_Res1(nn.Module):
    """
    stem block for Inception-ResNet-v1
    """

    def __init__(self):
        super(Stem_Res1, self).__init__()
        self.stem = nn.Sequential(
            BN_Conv2d(1, 32, (3, 3), 2, 0, bias=False),
            BN_Conv2d(32, 32, (3, 3), 1, 0, bias=False),
            BN_Conv2d(32, 64, (3, 3), 1, 0, bias=False),
            nn.MaxPool2d((3, 3), 2, 0),
            BN_Conv2d(64, 80, (1, 1), 1, 0, bias=False),
            BN_Conv2d(80, 192, (3, 3), 1, 0, bias=False),
            BN_Conv2d(192, 256, (3, 3), 2, 0, bias=False)
        )

    def forward(self, x):
        return self.stem(x)


class Stem_v4_Res2(nn.Module):
    """
    stem block for Inception-v4 and Inception-RestNet-v2
    """

    def __init__(self):
        super(Stem_v4_Res2, self).__init__()
        self.step1 = nn.Sequential(
            BN_Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            BN_Conv2d(32, 32, (3, 3), 1, 0, bias=False),
            BN_Conv2d(32, 64, (3, 3), 1, 1, bias=False)
        )
        self.step2_pool = nn.MaxPool2d((3, 3), 2, 0)
        self.step2_conv = BN_Conv2d(64, 96, (3, 3), 2, 0, bias=False)
        self.step3_1 = nn.Sequential(
            BN_Conv2d(160, 64, (1, 1), 1, 0, bias=False),
            BN_Conv2d(64, 96, (3, 3), 1, 0, bias=False)
        )
        self.step3_2 = nn.Sequential(
            BN_Conv2d(160, 64, (1, 1), 1, 0, bias=False),
            BN_Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(64, 96, (3, 3), 1, 0, bias=False)
        )
        self.step4_pool = nn.MaxPool2d((3, 3), 2, 0)
        self.step4_conv = BN_Conv2d(192, 192, (3, 3), 2, 0, bias=False)

    def forward(self, x):
        # pdb.set_trace()
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)  # 第一次拼接
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)  # 第二次拼接
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        out = torch.cat((tmp1, tmp2), 1)  # 第三次拼接
        return out


class Inception(nn.Module):
    """
    implementation of Inception-v4
    """

    def __init__(self, flatten_dim=1536, spkVec_dim=256):
        super(Inception, self).__init__()
        self.stem = Stem_v4_Res2()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()

        self.fl = nn.Sequential()
        self.fl.add_module('Linear_last', nn.Linear(flatten_dim, spkVec_dim))
        self.fl.add_module('ReLU_last', nn.ReLU(inplace=True))
        self.fl.add_module('bn_last', nn.BatchNorm1d(spkVec_dim))

        # if self.version == "v4":
        # self.fc = nn.Linear(1536, cvector_dim)
        # elif self.version == "res1":
        # self.fc = nn.Linear(1792, cvector_dim)
        # else:
        # self.fc = nn.Linear(2144, cvector_dim)

    def __make_inception_A(self):
        layers = []
        layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96, 96))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        return Reduction_A(384, 192, 224, 256, 384)  # 1024

    def __make_inception_B(self):
        layers = []
        for _ in range(7):
            layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                      192, 192, 224, 224, 256))  # 1024
        return nn.Sequential(*layers)

    def __make_reduction_B(self):
        return Reduction_B_v4(1024, 192, 192, 256, 256, 320, 320)  # 1536

    def __make_inception_C(self):
        layers = []
        for _ in range(3):
            layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        return nn.Sequential(*layers)

    def forward(self, x):
        # pdb.set_trace()
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        # pdb.set_trace()

        out = self.stem(x)
        out = self.inception_A(out)

        out = self.Reduction_A(out)

        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)
        # pdb.set_trace()
        out = F.avg_pool2d(out, (out.shape[2], out.shape[3]))
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        # pdb.set_trace()

        out = self.fl(out)

        return out
