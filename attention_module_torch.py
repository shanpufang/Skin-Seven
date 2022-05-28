from torch._C import import_ir_module
import torch.nn as nn
import torch
from torch.nn import functional as F
import get_dct_weight

class SE(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        return x



class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x



class MSCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MSCA, self).__init__()
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(...))
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.sum(x*self.pre_computed_dct_weights, dim = [2,3])
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.my_maxpool1 = nn.MaxPool2d(3)
        self.my_maxpool2 = nn.MaxPool2d(5)
        self.my_maxpool3 = nn.MaxPool2d(7)
        
        self.sigmoid = nn.Sigmoid()
-
    def forward(self, x):
        b,c,h,w = x.shape
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        
        temp1 = self.my_maxpool1(x).view(b,c,-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        temp2 = self.my_maxpool2(x).view(b,c,-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        temp3 = self.my_maxpool3(x).view(b,c,-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        my_out = self.fc2(self.relu1(self.fc1((temp1 + temp2 + temp3) / 3)))

        out = avg_out + max_out + my_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self,input_channel,ratio = 16):
        super().__init__()
        self.ca = ChannelAttention(input_channel,ratio)
        self.sa = SpatialAttention()
    
    def forward(self,x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


def attach_attention_module(attention_module, input_channel):
    if attention_module == 'SE':
        net = SE(ch_in)
    elif attention_module == 'ECA':
        net = ECA(channel)
    elif attention_module == 'MSCA':
        net = MSCA(channel)
    elif attention_module == 'CBAM': 
        net = CBAM(input_channel)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net
