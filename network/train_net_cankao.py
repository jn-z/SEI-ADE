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
from network.network_componet import CNN, fullyConnect, ArcMarginProduct,AddMarginProduct,SphereMarginProduct,MHELossFun
from models_1D import resnet50
from pytorch_revgrad import RevGrad
import pdb
class TrainModule(nn.Module):
    def __init__(self, args):

        super(TrainModule, self).__init__()
        self.args = args

        self.layer = nn.Sequential()
        self.layer.add_module('resnet', resnet50(self.args.cvector_dim))
        # self.layer.add_module('xvector', CNN())
        # self.layer.add_module('header', nn.ModuleList([fullyConnect(target_num=nspeakers, spkVec_dim=self.args.cvector_dim) \
        #   for nspeakers in self.args.headers]))
        self.marginType = args.marginType
        if args.marginType == 'sphere':
            self.layer.add_module('header', nn.ModuleList(
                [SphereMarginProduct(target_num=self.args.spk_num, spkVec_dim=self.args.cvector_dim, m=args.cemargin)]))
        elif args.marginType == 'arc':
            self.layer.add_module('header', nn.ModuleList([ArcMarginProduct(target_num=self.args.spk_num,
                                                                            spkVec_dim=self.args.cvector_dim,
                                                                            m=args.cemargin, s=args.cescale)]))
        elif args.marginType == 'add':
            self.layer.add_module('header', nn.ModuleList([AddMarginProduct(target_num=self.args.spk_num,
                                                                            spkVec_dim=self.args.cvector_dim,
                                                                            m=args.cemargin, s=args.cescale)]))
        else:
            self.layer.add_module('header', nn.ModuleList(
                [fullyConnect(target_num=self.args.spk_num, spkVec_dim=self.args.cvector_dim)]))
        self.revlayer = torch.nn.Sequential(copy.deepcopy(self.layer), RevGrad(alpha=args.alpha_parameter))
    # def forward(self, src, state_lab, total_fea_frames, npart,lam,ss):
    def forward(self, cpc_src, ss_src, state_lab, npart, is_train=True, is_tar=True):
        cpc_feature = cpc_src
        ss_feature = ss_src
        # anchor data
        # pdb.set_trace()
        anchor_sv = self.layer.resnet(cpc_feature)
        anchor_ss = self.revlayer.resnet(ss_feature)
        if self.marginType == 'softmax':
            tar = self.layer.header[npart](anchor_sv)
            ss_tar = self.revlayer.header[npart](anchor_ss)
        else:
            tar, we = self.layer.header[npart](anchor_sv, state_lab)
            ss_tar, ss_we = self.revlayer.header[npart](anchor_ss, state_lab)
        if is_train:
            tar_select_new = torch.gather(tar, 1, state_lab)
            ce_loss = -torch.log(tar_select_new + pow(10.0, -8))
            ss_tar_select_new = torch.gather(ss_tar, 1, state_lab)
            ss_loss = -torch.log(ss_tar_select_new + pow(10.0, -8))
            return anchor_sv, anchor_ss, tar_select_new, ss_tar_select_new, ce_loss, ss_loss, tar, ss_tar
        else:
            if is_tar:
                return tar, ss_tar
            else:
                return anchor_sv, anchor_ss
