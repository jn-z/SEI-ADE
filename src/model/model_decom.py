import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch.nn.init as init


def xavier(param):
    init.xavier_uniform(param)

class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.conv =nn.Conv1d(inChannels,growthRate,kernel_size=1,padding=0, bias=True)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class SingleBlock(nn.Module):
    def __init__(self, inChannels, growthRate, nDenselayer):
        super(SingleBlock, self).__init__()
        self.block = self._make_dense(inChannels, growthRate, nDenselayer)

    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class FDCN(nn.Module):
    def __init__(self, inChannels, growthRate, nDenselayer, nBlock):
        super(FDCN, self).__init__()

        self.conv1 = nn.Conv1d(1, growthRate, kernel_size=3, padding=1, bias=True)

        inChannels = growthRate

        self.denseblock = self._make_block(inChannels, growthRate, nDenselayer, nBlock)
        inChannels += growthRate * nDenselayer * nBlock

        self.Bottleneck = nn.Conv1d(in_channels=inChannels, out_channels=128, kernel_size=1, stride=4, padding=0, bias=True)

        self.convt1 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1,
                                         bias=True)

        self.convt2 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1,
                                         bias=True)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=2, kernel_size=3, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_block(self, inChannels, growthRate, nDenselayer, nBlock):
        blocks = []
        for i in range(int(nBlock)):
            blocks.append(SingleBlock(inChannels, growthRate, nDenselayer))
            inChannels += growthRate * nDenselayer
        return nn.Sequential(*blocks)

    def forward(self, x):

        out = F.relu(self.conv1(x))  #torch.Size([64, 6, 13200])
        out = self.denseblock(out)   #torch.Size([64, 30, 13200])
        out = self.Bottleneck(out)  #torch.Size([64, 128, 3300])
        out = self.convt1(out) #torch.Size([64, 128, 6600])
        out = self.convt2(out) # torch.Size([64, 128, 13200])
        HR = self.conv2(out) #torch.Size([64, 3, 13200])
        return HR



class Resblock(nn.Module):
    def __init__(self,channels,scale):
        super(Resblock,self).__init__()
        self.conv1 = nn.Conv1d(channels,channels,kernel_size=6,padding=4)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels,channels,kernel_size=6,padding=4)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self,x,scale):
        pdb.set_trace()
        Res = self.conv1(x)
        Res = self.bn1(Res)
        Res = self.relu(Res)
        Res = self.conv2(Res)
        Res = self.bn2(Res)
        #sacle = 0.15
        return x + Res*scale

def _pharse_shift(I,r):
    bsize,a,b,c = I.get_shape().as_list()
    bsize = torch._shape_as_tensor(I)[0]
    X = torch.reshape(I,(bsize,a,c,r,r))
    X = torch.transpose(X,(0,1,2,4,3))
    X = torch.split(X,a,1)
    X = torch.cat([torch.squeeze(x,dim=1) for x in X],2)
    X = torch.split(X,c,1)
    X = torch.cat([torch.squeeze(x,dim=1) for x in X],2)
    return torch.reshape(X,(bsize,a*c,r*r,1))


class up_sample(nn.Module):
    def __init__(self,channels):
        super(up_sample,self).__init__()

        self.up_block1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up_block21 = nn.Sequential(
            nn.Conv1d(channels,2,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.up_block22 = nn.Sequential(
            nn.Conv1d(channels,3,kernel_size=3,padding=1),
            nn.ReLU()
        )

    def _pharse_shift(self,I, r):
        a,b,c = I.size()
        New = torch.zeros(a,b//r,c*r)
        for i in range(a):
            for j in range(c*r):
                New[i,0,j] = I[i,j%r,j//r]
        return New

    def forward(self,x,sc):
        assert sc in [2, 3]
        if sc == 2:
            x = self.up_block1(x)

            x = self.up_block21(x)

            #x = self._pharse_shift(x, 2)

        elif sc == 3:
            x = self.up_block1(x)

            x = self.up_block22(x)

            #x = self._pharse_shift(x, 3)

        return x



class FDRN(nn.Module):
    def __init__(self):

        super(FDRN, self).__init__()

        self.conv_fw_1 = nn.Sequential(
            nn.Conv1d(1,256,kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.res_fw = Resblock(256,0.1)
        self.Up = up_sample(256)
        self.ReLu =nn.ReLU()
        #self.up_fw = upsample(x,scale=2,channels=256)
        #s

    def forward(self,x):

        x = self.conv_fw_1(x) #torch.Size([64, 256, 13200])
        for i in range(32):
            x = self.res_fw(x,0.15)
        #x = self.up_fw(x,2,256)
        x = self.Up(x,3)
        x = self.ReLu(x)

        return x