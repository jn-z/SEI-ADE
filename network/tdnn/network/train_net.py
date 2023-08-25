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
from network.network_componet import CNN, fullyConnect

class TrainModule(nn.Module):
    def __init__(self, args):
        super(TrainModule, self).__init__()
        self.args = args

        self.layer = nn.Sequential()
        self.layer.add_module('cnn', CNN())
        self.layer.add_module('header', nn.ModuleList([fullyConnect(target_num=nspeakers, spkVec_dim=self.args.cvector_dim) \
            for nspeakers in self.args.headers]))

    def forward(self, src, state_lab, lmcl_margin, total_fea_frames, npart):
        anchor_data = src
        # pos_data = src[:,int(total_fea_frames/3):int(total_fea_frames*2/3),:]
        # neg_data = src[:,int(total_fea_frames*2/3):,:]

        # batch_size, lengths, fea_dim = anchor_data.size()

        # anchor_data = anchor_data.permute(0,2,1).contiguous()
        # anchor_data = anchor_data.view(batch_size,1,fea_dim,-1)

        # pos_data = pos_data.permute(0,2,1).contiguous()
        # pos_data = pos_data.view(batch_size,1,fea_dim,-1)

        # neg_data = neg_data.permute(0,2,1).contiguous()
        # neg_data = neg_data.view(batch_size,1,fea_dim,-1)
     
        #anchor data
        # anchor_cnn_out = self.layer.resnet(anchor_data)
        # anchor_cnn_out = anchor_cnn_out.reshape(batch_size, -1 , lengths)
        # anchor_cnn_out = anchor_cnn_out.permute(0,2,1).contiguous()
        # anchor_sv = self.layer.xvector(anchor_cnn_out)
        # pdb.set_trace()
        anchor_sv = self.layer.cnn(anchor_data)

        tar = self.layer.header[npart](anchor_sv)
        tar_select_new = torch.gather(tar, 1, state_lab)
        ce_loss = -torch.log(tar_select_new+pow(10.0,-8))
        predict = tar.max(dim=1)[1]
        
        return anchor_sv,predict,ce_loss
        # with torch.no_grad():
        #     #positive data
        #     pos_cnn_out = self.layer.resnet(pos_data)
        #     pos_cnn_out = pos_cnn_out.reshape(batch_size, -1 , lengths)
        #     pos_cnn_out = pos_cnn_out.permute(0,2,1).contiguous()
        #     pos_sv = self.layer.xvector(pos_cnn_out)
    
        #     #negative data
        #     neg_cnn_out = self.layer.resnet(neg_data)
        #     neg_cnn_out = neg_cnn_out.reshape(batch_size, -1 , lengths)
        #     neg_cnn_out = neg_cnn_out.permute(0,2,1).contiguous()
        #     neg_sv = self.layer.xvector(neg_cnn_out)

        # CosineDistance = nn.CosineSimilarity(dim=1, eps=1e-6)
        # pos_distance_cosine = CosineDistance(anchor_sv, pos_sv)
        # neg_distance_cosine = CosineDistance(anchor_sv, neg_sv)
            
        # margin_pos_neg = 0.2
        # ve_loss = neg_distance_cosine + margin_pos_neg - pos_distance_cosine
        # ve_loss = F.relu(ve_loss)
        # ve_loss = ve_loss.view(anchor_sv.size(0), -1)

        # return anchor_sv, predict, ce_loss, ve_loss
