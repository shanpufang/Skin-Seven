from numpy.core.fromnumeric import mean, std
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import Dataset as Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as F
import random
import os
import numpy as np
import csv
from torch.utils.data import DataLoader as DataLoader

DEVICE = 'cuda'

class train_dataset(Dataset):
    def __init__(self, image_dir,image_label,means,stds,train = True,pre_train = False):
        super(train_dataset,self).__init__()
        self.image_dir = image_dir
        self.image_label = image_label
        self.length = len(self.image_dir)
        self.train = train
        self.means = means
        self.stds = stds
        self.pre_train = pre_train
    def __len__(self):
        return self.length
    
    def __getitem__(self,index):  # pad each image to 640 * 640
        image_dir = self.image_dir[index]
        label = self.image_label[index]
        #image = jpeg.JPEG(image_dir).decode()
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = read_image(image_dir)
        if not self.pre_train:
            image = (image - self.means) / self.stds
        else:
            image = image / 255
            image = (image - self.means) / self.stds
        resize_ratio = 0.75
        image = image.float()
        
        if image.shape[1] >= image.shape[2]:
            new_image = torch.zeros(3,600,450)
        else:
            new_image = torch.zeros(3,450,600)
        if (image.shape[1] != 450 or image.shape[2] != 600) and (image.shape[1] != 600 or image.shape[2] != 450):
            h,w = image.shape[1:3]
            max_original_size = max(h,w)
            min_original_size = min(h,w)
            max_size = 600
            size = 450
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
            
            if h < w:
                ratio = size / h
                new_shape = [size,int(w * ratio)]    
            else:
                ratio = size / w
                new_shape = [int(h * ratio), size]
            
            image = transform.Resize(new_shape)(image)
            
            #new_image = torch.zeros(3,new_shape[0],new_shape[1])
        new_image[:,:image.shape[1],:image.shape[2]] = image
        image = new_image
        image = transform.Resize([int(image.shape[1]  * resize_ratio), int(image.shape[2] * resize_ratio)])(image)
        
        if self.train:
            horizential, vertical = np.random.randint(2, size = 2)
            if horizential:
                image = torch.flip(image,[2])
            if vertical:
                image = torch.flip(image,[1])
            #image = transform.RandomCrop(224)(image)
            return image, label
        else:
            return image, label
    
    def get_img_info(self, index):
        image_dir = self.image_dir[index]
        image = read_image(image_dir)
        height = image.shape[1]
        width = image.shape[2]
        return height,width




if __name__=="__main__":
    train_dataset = train_dataset(root ='data_dir')
    train_loader = DataLoader(train_dataset)
    for i,data in enumerate(train_loader):
        print(1)
