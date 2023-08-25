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
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.feature_maps[1], layers[0], stride=1)
        self.layer2 = self._make_layer(self.feature_maps[2], layers[1], stride=2)
        self.layer3 = self._make_layer(self.feature_maps[3], layers[2], stride=2)
        self.layer4 = self._make_layer(self.feature_maps[4], layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(flatten_dim, spkVec_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(spkVec_dim)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion


        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        #pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #x = self.maxpool(x)

        x_1 = self.layer1(x)
        
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # mean_x_4 = torch.mean(x_4,dim=2)
        # var_x_4 = torch.var(x_4,dim=2)
        # # std_x_5=torch.std(x_5,dim=1,unbiased=False)
        # x_5 = torch.cat((mean_x_4,torch.sqrt(var_x_4+0.00001)),dim=2)
        # x_5 = x_5.contiguous().view(x.size(0), -1)
        x = self.avgpool(x_4)
        x = x.view(x.size(0), -1)
        # import pdb
        #pdb.set_trace()
        x_6 = self.fc(x_5)
        x_6 = self.relu2(x_6)
        speakerVector = self.bn2(x_6)

        return speakerVector  

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=1, output_nc=24, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
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
        assert(n_blocks >= 0)
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
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

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
    def __init__(self,input_dim=64,output_dim=512,context_size=5,stride=1,dilation=1,batch_norm=True):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation        
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
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
        x = F.unfold(x,(self.context_size, self.input_dim),stride=(1,self.input_dim),dilation=(self.dilation,1))
        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        if self.batch_norm:
            x = x.transpose(1,2).contiguous()
            x = self.bn(x)
            x = x.transpose(1,2).contiguous()

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
        mean_x_5=torch.mean(x_0,dim=1)
        var_x_5=torch.var(x_0,dim=1)
        # std_x_5=torch.std(x_5,dim=1,unbiased=False)
        statistic_x_5=torch.cat((mean_x_5,torch.sqrt(var_x_5+0.00001)),dim=1)
        x_6 = self.layer6(statistic_x_5)         
        speakerVector = self.layer7(x_6)         
        return speakerVector                   

class fullyConnect(nn.Module):
    def __init__(self, target_num=10000,spkVec_dim=256):
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
        lastBias   = self.layer1.linear1.bias.unsqueeze(0)

        pad_cols     = self.pad_cols.repeat(dropedSpeakerVector.size(0), 1)
        #test_index_margin = torch.nonzero(lmcl_margin)

        cattedWeight = torch.cat((lastWeight.t(), lastBias), 0)
        cattedSpeakerVector = torch.cat((dropedSpeakerVector, pad_cols), 1)
        # debug
        '''
        hidden_test = torch.matmul(cattedSpeakerVector, cattedWeight)
        hidden_test = torch.matmul(dropedSpeakerVector, lastWeight.t()) + lastBias.expand(dropedSpeakerVector.size(0), lastBias.size(1))
        delta_hidden = hidden_test - hiddenVec
        '''
        # L2 norm in the SpeakerVector dimension
        catted_weight_length          = torch.norm(cattedWeight, 2, 0)
        catted_speakerVector_length   = torch.norm(cattedSpeakerVector, 2, 1)

        catted_weight_norm         = catted_weight_length.contiguous().view(1, -1).expand(cattedWeight.size())
        catted_speakerVector_norm1  = catted_speakerVector_length.contiguous().view(-1, 1).expand(cattedSpeakerVector.size())
        catted_speakerVector_norm2  = catted_speakerVector_length.contiguous().view(-1, 1).expand(cattedSpeakerVector.size(0), cattedWeight.size(1))
       
        normedWeight        = cattedWeight/catted_weight_norm
        normedSpeakerVector = cattedSpeakerVector/catted_speakerVector_norm1
        pscore = catted_speakerVector_norm2*(torch.matmul(normedSpeakerVector, normedWeight)-lmcl_margin)

        tar = F.softmax(pscore, dim=1)
        return speakerVector,tar





class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, target_num=10000, spkVec_dim=512, s=64.0, m=0.25,is_anneal=False, easy_margin=False):
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
            self.lamb = max(self.lambda_min, self.lambda_base * (1 + self.lambda_gamma * self.iter) ** (-1 * self.lambda_power))
            phi = (self.lamb * cosine + phi) / (1 + self.lamb)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        tar = F.softmax(output, dim=1)
        # NormOfFeature = torch.norm(input, 2, 1)
        # output *= NormOfFeature.view(-1, 1)
        return tar,weight



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
        return tar,weight


class SphereMarginProduct(nn.Module):
    def __init__(self,target_num=10000, spkVec_dim=512, m=2):
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
        return tar,weight

class MHELossFun(nn.Module):
    def __init__(self, mhe_type = 1,mhe_lambda=0.01):
        super(MHELossFun, self).__init__()
        self.mhe_lambda = mhe_lambda
        self.mhe_type = mhe_type
    def forward(self, labels,weight):
        trans_w = F.normalize(weight)#[100,512]
        w_norm = trans_w.t() #[512,100]
        num_classes = trans_w.shape[0]
                    
        # np.savetxt("./weight.txt", weight.t().cpu().detach().numpy(),fmt='%f',delimiter=',')
        # np.savetxt("./label.txt", labels.squeeze(1).cpu().detach().numpy(),fmt='%f',delimiter=',')
        if self.mhe_type == 1:
            # pdb.set_trace()
            sel_w = torch.index_select(trans_w, 0, labels.squeeze(1))
            batch_size = sel_w.shape[0]
            # Version 1
            # The inner product may be faster than the element-wise operations.
            dist = (2.0 - 2.0 * torch.matmul(sel_w, w_norm))#[10,100]
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
