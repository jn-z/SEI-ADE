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
from network.network_componet import CNN, fullyConnect, ResnetGenerator,ArcMarginProduct,AddMarginProduct,SphereMarginProduct
import pdb
class TrainModule(nn.Module):
    def __init__(self, args):

        super(TrainModule, self).__init__()
        self.args = args
        # pdb.set_trace()
        self.layer = nn.Sequential()
        self.layer.add_module('resnet', ResnetGenerator())
        self.layer.add_module('xvector', CNN())
        self.marginType =  args.marginType
        if args.marginType == 'sphere':
            self.layer.add_module('header', nn.ModuleList([SphereMarginProduct(target_num=nspeakers, spkVec_dim=self.args.cvector_dim,m = args.cemargin) \
                for nspeakers in self.args.headers]))
        elif args.marginType == 'arc':
            self.layer.add_module('header', nn.ModuleList([ArcMarginProduct(target_num=nspeakers, spkVec_dim=self.args.cvector_dim,m = args.cemargin,s = args.cescale) \
                for nspeakers in self.args.headers]))
        elif args.marginType == 'add':
            self.layer.add_module('header', nn.ModuleList([AddMarginProduct(target_num=nspeakers, spkVec_dim=self.args.cvector_dim,m = args.cemargin,s = args.cescale) \
                for nspeakers in self.args.headers]))
        else:
            self.layer.add_module('header', nn.ModuleList([fullyConnect(target_num=nspeakers, spkVec_dim=self.args.cvector_dim) \
                for nspeakers in self.args.headers]))
        
        # self.layer.add_module('header2', nn.ModuleList([fullyConnect(target_num=nspeakers, spkVec_dim=self.args.cvector_dim) \
        #         for nspeakers in self.args.headers]))

    def forward(self, src, state_lab, total_fea_frames, npart):
        anchor_data = src
        # anchor_data = src[:,0:int(total_fea_frames/3),:]
        batch_size, lengths, fea_dim = anchor_data.size()

        anchor_data = anchor_data.permute(0,2,1).contiguous()
        anchor_data = anchor_data.view(batch_size,1,fea_dim,-1)
     
        #anchor data
        anchor_cnn_out = self.layer.resnet(anchor_data)
        anchor_cnn_out = anchor_cnn_out.reshape(batch_size, -1 , lengths)
        anchor_cnn_out = anchor_cnn_out.permute(0,2,1).contiguous()
        anchor_sv = self.layer.xvector(anchor_cnn_out)
        #pdb.set_trace()
        #tar = self.layer.header[npart](anchor_sv)
        if  self.marginType == 'softmax':
            tar = self.layer.header[npart](anchor_sv)
        else:
            tar = self.layer.header[npart](anchor_sv,state_lab)
        tar_select_new = torch.gather(tar, 1, state_lab)
        ce_loss = -torch.log(tar_select_new+pow(10.0,-8))
        # tar2 = self.layer.header2[npart](anchor_sv)
        # tar_select_new2 = torch.gather(tar2, 1, state_lab)
        # ce_loss2 = -torch.log(tar_select_new2+pow(10.0,-8))
        # ce_loss = -torch.log(tar_select_new)
        predict = tar.max(dim=1)[1]
        
        return anchor_sv,predict,ce_loss
        # # predict2 = tar2.max(dim=1)[1]
        # # pdb.set_trace()
        
        # ve_loss_data = {}
        # ve_loss_data["dist"] = calc_cdist(anchor_sv,anchor_sv)
        # ve_loss_data["pids"] = torch.squeeze(state_lab)
        # margin_pos_neg = 'soft'
        # #alpha = adjust_alpha_rate(loss_fn, t)
        # bht = BatchHard(margin_pos_neg)
        # ve_losses = bht.forward(ve_loss_data["dist"],ve_loss_data["pids"])
        # # pdb.set_trace()
        # # ve_loss = F.relu(ve_loss)
        # ve_loss = ve_losses.view(anchor_sv.size(0), -1)

        # return anchor_sv, predict, ce_loss, ve_loss
