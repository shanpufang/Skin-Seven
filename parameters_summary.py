import torch
from my_summary import summary
from read_data3 import train_dataset

from densenet_121 import densenet121
from resnet import resnet50,resnet101,resnext50
from vggnet import vgg16,vgg19
from inception import InceptionV3
from se_resnet import se_resnet50,se_resnet101,se_resnext50,se_resnext101
from densenet import my_densenet

from flops_compute import profile

def test1():
    names = ['Vgg16','Vgg19',
            'ResNeXt-50','ResNet-101','ResNet-50','SE-ResNeXt50','SE-ResNeXt101','SE-ResNet50','SE-ResNet101','InceptionV3','DenseNet121',
            'ours']
    for j,name in enumerate(net_path):
        if j == 0:
            model = vgg16()
        elif j == 1:
            model = vgg19()
        elif j == 2:
            model = resnext50()
        elif j == 3:
            model = resnet101()
        elif j == 4:
            model = resnet50()
        elif j == 5:
            model = se_resnext50()
        elif j == 6:
            model = se_resnext101()
        elif j == 7:
            model = se_resnet50()
        elif j == 8:
            model = se_resnet101()
        elif j == 9:
            model = InceptionV3()
        elif j == 10:
            model = densenet121()
        else:
            model = my_densenet(attention_module = 'CBAM',pre_train = True)  
        nums = summary(model,input_size = (3,337,450))
        test_image = torch.randn([1,3,337,450])
        flops,params = profile(model,inputs = test_image)
        with open('./record.txt','a') as file_handle:   # .txt可以不自己新建,代码会自动新建
            file_handle.write(names[j] + '(' + str(nums.item()) + ')'+ 'flops' + str(flops) +'\n')    
    file_handle.close()    
        


net_path = ['vgg16_net_path',
            'vgg19_net_path',
            'resnext50_net_path',
            'resnet101_net_path',
            'resnet50_net_path',
            'se_resnext50_net_path',
            'se_resnext101_net_path',
            'se_resnet50_net_path',
            'se_resnet101_net_path',
            'InceptionV3_net_path',
            'densenet121_net_path',
            'mydensenet_net_path'
            ]

test_image = torch.randn([1,3,337,450])
model = mydensenet()
flops,_ = profile(model,inputs = test_image)
print(flops)
test1()
