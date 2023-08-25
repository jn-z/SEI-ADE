import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
import torch

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
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('tdnn1', TDNN(input_dim=64, output_dim=512, context_size=5, dilation=1))
        self.layer2 = nn.Sequential()
        self.layer2.add_module('tdnn2', TDNN(input_dim=512, output_dim=1536, context_size=3, dilation=2))
        self.layer3 = nn.Sequential()
        self.layer3.add_module('tdnn3', TDNN(input_dim=1536, output_dim=512, context_size=3, dilation=3))
        self.layer4 = nn.Sequential()
        self.layer4.add_module('tdnn4', TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1))
        self.layer5 = nn.Sequential()
        self.layer5.add_module('tdnn5', TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1))
        
        self.layer6 = nn.Sequential()
        self.layer6.add_module('linear6', nn.Linear(3000, 512))
        self.layer6.add_module('tdnn6_relu', torch.nn.ReLU())
        self.layer6.add_module('tdnn6_bn', nn.BatchNorm1d(512))        
        self.layer7 = nn.Sequential()
        # self.layer7.add_module('dropout7', nn.Dropout(p=0.2))
        self.layer7.add_module('linear7', nn.Linear(512, 512))
        self.layer7.add_module('tdnn7_relu', torch.nn.ReLU())
        self.layer7.add_module('tdnn7_bn', nn.BatchNorm1d(512))           
        
        # self.initial_parameters()

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x_0):
        # pdb.set_trace()
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        mean_x_5=torch.mean(x_5,dim=1)
        var_x_5=torch.var(x_5,dim=1)
        # std_x_5=torch.std(x_5,dim=1,unbiased=False)
        statistic_x_5=torch.cat((mean_x_5,torch.sqrt(var_x_5+0.00001)),dim=1)
        x_6 = self.layer6(statistic_x_5)         
        speakerVector = self.layer7(x_6)         
        return speakerVector        
        

class fullyConnect(nn.Module):
    def __init__(self, target_num=10000,spkVec_dim=512):
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