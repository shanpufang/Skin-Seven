from numpy.core.fromnumeric import mean, std
import torch
import torch.nn as nn
from collections import OrderedDict
from attention_module_torch import attach_attention_module
from torchvision.models import densenet121 as densenet
import math


class my_densenet121(nn.Module):
    def __init__(self,growth_rate = 32, block_config = (6,12,24,16),
                 bn_size = 4, theta = 0.5, num_classes = 7,attention_module = 'CBAM',pre_train = True):
        super(my_densenet121,self).__init__()
        densenet121 = densenet(pretrained = pre_train)
        num_init_feature = 2 * growth_rate
        num_feature = num_init_feature
        self.pre_train = pre_train
        self.features = nn.Sequential()
        names = {0:'conv0',1:'norm0',2:'relu0',3:'pool0',4:'denseblock1',5:'transition1',
                 6:'denseblock2',7:'transition2',8:'denseblock3',9:'transition3',
                 10:'denseblock4',11:'norm5'}
        j = 0
        densenet_layers = list(densenet121.children())
        num_layers = 0
        for i in range(len(names)):
            num_layers = block_config[j]
            if i < 4:
                layers = densenet_layers[0][i]
                if pre_train:
                    for p in layers.parameters():
                        p.require_grad = False
                self.features.add_module(names[i],layers)
            else:
                layers = densenet_layers[0][i]
                self.features.add_module(names[i],layers)
            if i == 4 or i == 6 or i == 8:
                num_feature = num_feature + growth_rate * num_layers
                if attention_module:
                    self.features.add_module('attention%d' % (j + 1),attach_attention_module(attention_module,num_feature))
                    #print(self.features)
                num_feature = int(num_feature * theta)
                j += 1
        num_feature = num_feature + growth_rate * num_layers 
        #self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.classifier = nn.Sequential()
        self.convs = nn.Sequential()
        
        for i in range(0):
            self.convs.add_module('conv%d' % (i + 1),nn.Conv2d(num_feature,256,3,1,1))
            self.convs.add_module('groupnorm%d' % (i + 1),nn.GroupNorm(32,256))
            self.convs.add_module('relu%d' % (i + 1),nn.ReLU(inplace=True))
            num_feature = 256
        
        self.classifier.add_module('conv',nn.Conv2d(num_feature,num_classes,3,1,1))
        prior = torch.tensor([1e-2,0.8,1e-2,1e-2,1e-2,1e-2,1e-2])
        
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.constant_(m.bias,0)
        #for m in self.classifier.modules():
            #if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight,mean = 0,std=0.01)
                #for i in range(num_classes):
                    #nn.init.constant_(m.bias[i], -torch.log((1 - prior[i]) / prior[i]))
        self.classifier.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
    
    def train(self,mode=True):
        
        #set module training mode, and frozen bn
        
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        if self.pre_train:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        else:
            print("INFO===>Didn't frozen BN")
            
    def forward(self, x):
        features = self.features(x)
        features = self.convs(features)
        #out = features.view(features.size(0), -1)
        out = self.classifier(features)
        out = torch.squeeze(out)
        return out


def my_densenet(num_classes=7,attention_module='CBAM', pre_train = True):
    return my_densenet121(num_classes=num_classes, attention_module=attention_module,pre_train=pre_train)

