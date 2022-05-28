import torch
import torch.nn as nn
from collections import OrderedDict
import math

class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,bias=True))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1,bias=True))

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1,bias=True)) # origin is flase
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, num_classes=7):
        super(DenseNet_BC, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        # 表示cifar-10
        if num_classes == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=7, stride=2,
                                    padding=3,bias=True)),#original bias = False
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))



        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            #if attention_module:
                #self.features.add_module('attention%d' % (i + 1),attach_attention_module(attention_module,num_feature))
                #print(self.features)
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(num_feature, num_classes)
        prior = torch.tensor([0.01,0.8,0.01,0.01,0.01,0.01,0.01])
        #nn.init.normal_(self.classifier.weight, std=0.001)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias == None:
                    continue
                #else:
                    #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                for i in range(num_classes):
                    nn.init.constant_(self.classifier.bias[i], -torch.log((1 - prior[i]) / prior[i]))
                #nn.init.constant_(self.classifier.bias,0)

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121(num_classes=7):
  return DenseNet_BC(
                  num_classes=num_classes, 
                  growth_rate=32,
                  block_config=[6,12,24,16])
densenet121.default_image_size = 224


def densenet161(num_classes=7):
  return DenseNet_BC(
                  num_classes=num_classes, 
                  growth_rate=48,
                  num_filters=96,
                  block_config=[6,12,36,24])
densenet161.default_image_size = 224


def densenet169(num_classes=7):
  return DenseNet_BC(
                  num_classes=num_classes, 
                  growth_rate=32,
                  block_config=[6,12,32,32])
densenet169.default_image_size = 224

def densenet201(num_classes=7):
  return DenseNet_BC(
                  num_classes=num_classes, 
                  growth_rate=32,
                  block_config=[6,12,48,32])
densenet201.default_image_size = 224
'''
def densenet_arg_scope(weight_decay=1e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1.1e-5,
                ):
  with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.avg_pool2d, slim.max_pool2d,
                       _conv_block, _global_avg_pool2d],
                      data_format=data_format):
    with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         activation_fn=None,
                         biases_initializer=None):
      with slim.arg_scope([slim.batch_norm],
                          scale=True,
                          decay=batch_norm_decay,
                          epsilon=batch_norm_epsilon) as scope:
        return scope
'''
