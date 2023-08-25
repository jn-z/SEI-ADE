from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb
import copy
from network.pytorch_revgrad import RevGrad
## PyTorch implementation of CDCK2, CDCK5, CDCK6, speaker classifier models
# CDCK2: base model from the paper 'Representation Learning with Contrastive Predictive Coding'
# CDCK5: CDCK2 with a different decoder
# CDCK6: CDCK2 with a shared encoder and double decoders
# SpkClassifier: a simple NN for speaker classification
def test_gradients_inverted():
    network = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 1))
    pdb.set_trace()
    revnetwork = torch.nn.Sequential(copy.deepcopy(network), RevGrad())
    pdb.set_trace()
    gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
    revgru = torch.nn.Sequential(copy.deepcopy(gru), RevGrad())
    inp = torch.randn(8,128, 512)
    outp = torch.randn(8,128, 256)
    gru(inp)
    revgru(inp)
    criterion = torch.nn.MSELoss()

    criterion(gru(inp), outp).backward()
    criterion(revgru(inp), outp).backward()
    print("dywfhjkj")
    assert all(
        (p1.grad == -p2.grad).all()
        for p1, p2 in zip(network.parameters(), revnetwork.parameters())
    )
class CDCK6(nn.Module):
    ''' CDCK2 with double decoder and a shared encoder '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK6, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru1 = nn.GRU(512, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk1  = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.gru2 = nn.GRU(512, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk2  = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru1 and gru2
        for layer_p in self.gru1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru1.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru2._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru2.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden1(self, batch_size): # initialize gru1
        #return torch.zeros(1, batch_size, 128).cuda()
        return torch.zeros(1, batch_size, 128)

    def init_hidden2(self, batch_size): # initialize gru2
        #return torch.zeros(1, batch_size, 128).cuda()
        return torch.zeros(1, batch_size, 128)

    def forward(self, x, x_reverse, hidden1, hidden2):
        batch = x.size()[0]
        nce = 0 # average over timestep and batch and gpus
        t_samples = torch.randint(self.seq_len/160-self.timestep, size=(1,)).long() # randomly pick time stamps. ONLY DO THIS ONCE FOR BOTH GRU.

        # first gru
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output1, hidden1 = self.gru1(forward_seq, hidden1) # output size e.g. 8*100*256
        c_t = output1[:,t_samples,:].view(batch, 128) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk1[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor

        # second gru
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x_reverse)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output2, hidden2 = self.gru2(forward_seq, hidden2) # output size e.g. 8*100*256
        c_t = output2[:,t_samples,:].view(batch, 128) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk2[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor

        nce /= -1.*batch*self.timestep
        nce /= 2. # over two grus
        accuracy = 1.*(correct1.item()+correct2.item())/(batch*2) # accuracy over batch and two grus
        #print(torch.cat((output1, output2), dim=2).shape)

        return accuracy, nce, hidden1, hidden2

    def predict(self, x, x_reverse, hidden1, hidden2):
        batch = x.size()[0]

        # first gru
        # input sequence is N*C*L, e.g. 8*1*20480
        z1 = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z1 = z1.transpose(1,2)
        output1, hidden1 = self.gru1(z1, hidden1) # output size e.g. 8*128*256

        # second gru
        z2 = self.encoder(x_reverse)
        z2 = z2.transpose(1,2)
        output2, hidden2 = self.gru2(z2, hidden2)

        return torch.cat((output1, output2), dim=2) # size (64, seq_len, 256)
        #return torch.cat((z1, z2), dim=2) # size (64, seq_len, 512*2)


class CDCK5(nn.Module):
    ''' CDCK2 with a different decoder '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK5, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential(  # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(40, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size):
        #return torch.zeros(2*1, batch_size, 40).cuda()
        return torch.zeros(2*1, batch_size, 40)

    def forward(self, x, hidden):

        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        #pdb.set_trace()
        z = z.transpose(1,2)
        #output, hidden = self.gru(z, hidden) # output size e.g. 8*100*40

        return z

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512

        z = z.transpose(1,2)
        #output, hidden = self.gru(z, hidden) # output size e.g. 8*128*40

        return z # return every frame
        #return output[:,-1,:], hidden # only return the last frame per utt


class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, seq_len,alpha_parameter):

        super(CDCK2, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.alpha_parameter = alpha_parameter
        self.layer = nn.Sequential()
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.revgru = nn.Sequential(copy.deepcopy(self.gru), RevGrad(alpha=self.alpha_parameter))
        self.revWk = nn.ModuleList([nn.Sequential(copy.deepcopy(self.Wk[i]), RevGrad(alpha=self.alpha_parameter)) for i in range(timestep)])
        self.revencoder = nn.Sequential(copy.deepcopy(self.encoder),RevGrad(alpha=self.alpha_parameter))

        #self.revlayer.add_module('revgrad')

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, 256).cuda()
        else: return torch.zeros(1, batch_size, 256)

    def forward(self, cpc_x, ss_x, hidden):
        batch = cpc_x.size()[0]

        t_samples = torch.randint(int(self.seq_len/40-self.timestep), size=(1,)).long() # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        #pdb.set_trace()
        z = self.encoder(cpc_x)

        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512

        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output, hidden_ori = self.gru(forward_seq) # output size e.g. 8*100*256
        c_t = output[:,t_samples,:].view(batch, 256) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512

        # for rev_x
        rev_z = self.revencoder(ss_x)
        rev_z = rev_z.transpose(1, 2)
        nce = 0  # average over timestep and batch
        rev_nce = 0
        rev_encode_samples = torch.empty((self.timestep, batch, 512)).float()  # e.g. size 12*8*512
        for i in np.arange(1, self.timestep + 1):
            rev_encode_samples[i - 1] = rev_z[:, t_samples + i, :].view(batch, 512)  # z_tk e.g. size 8*512
        #pdb.set_trace()
        rev_forward_seq = rev_z[:, :t_samples + 1, :]  # e.g. size 8*100*512

        rev_output, hidden_rev = self.revgru(rev_forward_seq)  # output size e.g. 8*100*256
        rev_c_t = rev_output[:, t_samples, :].view(batch, 256)  # c_t e.g. size 8*256
        rev_pred = torch.empty((self.timestep, batch, 512)).float()  # e.g. size 12*8*512

        for i in np.arange(0, self.timestep):
            rev_linear = self.revWk[i]
            rev_pred[i] = rev_linear(rev_c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            #pdb.set_trace()
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor

            rev_total = torch.mm(rev_encode_samples[i], torch.transpose(rev_pred[i], 0, 1))  # e.g. size 8*8
            rev_correct = torch.sum(torch.eq(torch.argmax(self.softmax(rev_total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            rev_nce += torch.sum(torch.diag(self.lsoftmax(rev_total)))  # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        rev_nce /= -1. * batch * self.timestep
        rev_accuracy = 1. * rev_correct.item() / batch

        output, hidden_ori = self.gru(z)
        ss_output, hidden_rev = self.revgru(rev_z)

        return output, ss_output, accuracy, nce, rev_accuracy, rev_nce, hidden

    def predict(self, cpc_x, ss_x, hidden):
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(cpc_x)
        rev_z = self.revencoder(ss_x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        rev_z = rev_z.transpose(1, 2)
        output, hidden = self.gru(z) # output size e.g. 8*128*256
        ss_output, rev_hidden = self.revgru(rev_z)

        return output, ss_output, hidden # return every frame


class SpkClassifier(nn.Module):
    ''' linear classifier '''
    def __init__(self, spk_num):

        super(SpkClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, spk_num)
            #nn.Linear(256, spk_num)
        )

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x):
        #import pdb
        #pdb.set_trace()
        x = self.classifier(x)

        return F.log_softmax(x, dim=-1)

