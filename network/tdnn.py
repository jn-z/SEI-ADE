import torch
import torch.nn as nn
import torch.nn.functional as F

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
