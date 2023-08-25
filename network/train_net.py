import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F
import numpy as np
import math
import copy
from torch.autograd import Variable
from module.utils import BatchEndParam
from network.network_componet import CNN, fullyConnect, ResNet, ArcMarginProduct,AddMarginProduct,SphereMarginProduct,MHELossFun,RNNnet, DNNnet, DenseNet, VGG, GoogLeNet, Lstm, AlexNet
from network.models_1D import resnet50
from network.pytorch_revgrad import RevGrad
import copy
import pdb
class TrainModule(nn.Module):
    def __init__(self, args):

        super(TrainModule, self).__init__()
        self.args = args

        self.Rlayer = nn.Sequential()
        self.Rlayer.add_module('resnet', ResNet(layers=[1, 1, 1, 0], flatten_dim=32768, spkVec_dim=self.args.cvector_dim))

        self.marginType = args.marginType
        if args.marginType == 'sphere':
            self.Rlayer.add_module('header', nn.ModuleList(
                [SphereMarginProduct(target_num=self.args.spk_num, spkVec_dim=self.args.cvector_dim, m=args.cemargin)]))
        elif args.marginType == 'arc':
            self.Rlayer.add_module('header', ArcMarginProduct(target_num=self.args.spk_num,
                                                                            spkVec_dim=self.args.cvector_dim,
                                                                            m=args.cemargin, s=args.cescale))
        elif args.marginType == 'add':
            self.Rlayer.add_module('header', AddMarginProduct(target_num=self.args.spk_num,
                                                                            spkVec_dim=self.args.cvector_dim,
                                                                            m=args.cemargin, s=args.cescale))
        else:
            self.Rlayer.add_module('header', fullyConnect(target_num=self.args.spk_num, spkVec_dim=self.args.cvector_dim))
        self.revResNet = nn.Sequential(copy.deepcopy(self.Rlayer.resnet),RevGrad(alpha=self.args.alpha_parameter))
        self.revheader = nn.Sequential(copy.deepcopy(self.Rlayer.header), RevGrad(alpha=self.args.alpha_parameter))

        #self.Rrevlayer.add_module('Rlayer', copy.deepcopy(self.Rlayer))
        #self.Rrevlayer.add_module('revgrad', )
    # def forward(self, src, state_lab, total_fea_frames, npart,lam,ss):
    def forward(self, cpc_src,  ss_src,   state_lab, is_train=True, is_tar=True):
        cpc_feature = cpc_src
        ss_feature = ss_src
        # anchor data

        anchor_sv = self.Rlayer.resnet(cpc_feature)
        anchor_ss = self.revResNet(ss_feature)
        if self.marginType == 'softmax':
            tar = self.Rlayer.header(anchor_sv)
            ss_tar = self.revheader(anchor_ss)
        else:
            ss_tar, ss_we = self.revheader(anchor_ss, state_lab)
            tar, we = self.Rlayer.header(anchor_sv, state_lab)
        if is_train:
            tar_select_new = torch.gather(tar, 1, state_lab)
            ce_loss = -torch.log(tar_select_new + pow(10.0, -8))
            ss_tar_select_new = torch.gather(ss_tar, 1, state_lab)
            ss_loss = -torch.log(ss_tar_select_new + pow(10.0, -8))
            return anchor_sv, anchor_ss,   tar_select_new, ss_tar_select_new,  ce_loss, ss_loss, tar, ss_tar
        else:
            if is_tar:
                return tar, ss_tar
            else:
                return anchor_sv, anchor_ss
