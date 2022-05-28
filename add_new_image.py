import numpy as np
import math
import torch
from torch._C import import_ir_module
import torch.nn as nn
import csv
import os

image_dir = 'add_data_dir'
image_classes = os.listdir(image_dir)
dict = {'MEL':['1','0','0','0','0','0','0'],'NV':['0','1','0','0','0','0','0'],'BCC':['0','0','1','0','0','0','0']
,'AKIEC':['0','0','0','1','0','0','0'],'BKL':['0','0','0','0','1','0','0'],'DF':['0','0','0','0','0','1','0'],'VASC':['0','0','0','0','0','0','1']}
f = open('csv_dir','a+',encoding='utf-8')
csv_writer = csv.writer(f)
#csv_writer.writerow(["image","MEL","NV","BCC","AKIEC","BKL","DF","VASC"])
for image_class in image_classes:
    for image_name in os.listdir(os.path.join(image_dir,image_class)):
        if image_name == 'C100g.jpg':
            print(1)
        image_name = [image_name.replace('.jpg','')]
        a = image_name + dict[image_class]
        csv_writer.writerow(image_name + dict[image_class])
f.close()
'''
a = torch.randn([2,3])
print(a)
mask = a > 0.5
print(mask)
b = a[mask]
print(b)
b = torch.mean(b)
print(b)
'''
