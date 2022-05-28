import bisect
import copy
import csv
from enum import EnumMeta
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interp
from sklearn.metrics import auc, roc_curve
from torch.cuda.amp import \
    GradScaler as GradScaler  # only supported in pytorch 1.6
from torch.cuda.amp import \
    autocast as autocast  # only supported in pytorch 1.6
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import BatchSampler
from torchvision.io import read_image
from torchvision.models.densenet import DenseNet
from tqdm import tqdm
import torch.nn.functional as F

from densenet_121 import densenet121
from resnet import resnet50,resnet101,resnext50
from vggnet import vgg16,vgg19
from inception import InceptionV3
from se_resnet import se_resnet50,se_resnet101,se_resnext50,se_resnext101
from densenet import my_densenet

from read_data3 import train_dataset
import time
from thop import profile

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()

    def start(self):
        self.preload()

    def preload(self):
        try:
            self.batch_imgs, self.labels = next(self.loader)
        except StopIteration:
            self.batch_imgs = None
            return
        with torch.cuda.stream(self.stream):
            self.batch_imgs = self.batch_imgs.cuda(non_blocking = True).half()
            self.labels = self.labels.cuda(non_blocking = True).int()
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.batch_imgs
        labels = self.labels
        self.preload()
        return image,labels

class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``

    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven
        self.groups = torch.unique(self.group_ids).sort(0)[0]
        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)

def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def _compute_aspect_ratios(set):
    aspect_ratios = []
    for i in range(len(set)):
        height,width = set.get_img_info(i)
        aspect_ratio = height / width
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios

def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = sampler.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def get_set(root,label_dir = 'csv_dir'):
    csv_dir = os.path.join(root,label_dir)
    train_set = []
    train_label = []
    val_set = []
    val_label = []
    image_dir = 'data_dir'
    means = torch.zeros(3)
    stds = torch.zeros(3)
    with open(csv_dir)as f:
        with tqdm(total = 10868) as bar:
            bar.set_description('Data Preparation')
            f_csv = csv.reader(f)
            i = 0
            for row in f_csv:
                if i == 0:
                    i+=1
                    bar.update(1)
                    continue
                file_name = row[0]
                image_label = np.array(row[1:],dtype=np.float)
                image_label = torch.from_numpy(image_label).half()
                image_name = file_name + '.jpg'
                image_name = os.path.join(root,image_dir,image_name)
                np.random.seed(i)
                train_num = np.random.uniform(0,1)
                if train_num < 0.8:   
                    image = read_image(image_name)
                    image.requires_grad = False
                    means += torch.mean(image.float(),dim = [1,2])
                    stds += torch.std(image.float(),dim = [1,2])
                    train_set.append(image_name)
                    train_label.append(image_label)
                else:
                    val_set.append(image_name)
                    val_label.append(image_label)
                bar.update(1)
                i+=1
    numbers = 0
    means = means.reshape(3,1,1) / len(train_set)
    stds = stds.reshape(3,1,1) / len(train_set)
    return [train_set,train_label],[val_set,val_label],numbers,means,stds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

trains,vals,numbers,means,stds = get_set(root ='data_dir')

testset = train_dataset(vals[0],vals[1],means,stds,False)
test_sampler = make_data_sampler(testset,True)
test_BatchSampler = make_batch_data_sampler(testset,test_sampler,1,BATCH_SIZE)
test_loader = DataLoader(testset, batch_sampler=test_BatchSampler, num_workers = 16, pin_memory = True)
Closs = nn.CrossEntropyLoss()
print('The dataset preparation is done')


def anti_onehot(labels):
    result = torch.zeros(labels.shape[0])
    result = torch.argmax(labels,dim=1)
    return result

def get_recall_precition(matrix,total_num):
    recall = []
    precition = []
    specificity = []
    for i in range(matrix.shape[0]):
        temp_class = matrix[i,i]
        #recall_compute =  torch.sum(matrix[i,:])
        precition_compute = torch.sum(matrix[:,i])
        negative = torch.sum(matrix) - torch.sum(matrix[i,:])
        TN = negative - torch.sum(matrix[:,i]) + matrix[i,i]
        specificity.append(float(TN / negative))
        recall.append(float(temp_class / total_num[i]))
        precition.append(float(temp_class / precition_compute))
    return recall, precition, specificity

def test1(testloader,testset):
    y_test = torch.zeros([len(net_path),testset.length,7])
    y_score = torch.zeros_like(y_test)
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
        model.load_state_dict(torch.load(name))
        model.eval()
        
        if j == len(net_path) - 1:
            means = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
            stds = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
            testset = train_dataset(vals[0],vals[1],means,stds,False,True)
            test_sampler = make_data_sampler(testset,True)
            test_BatchSampler = make_batch_data_sampler(testset,test_sampler,1,BATCH_SIZE)
            testloader = DataLoader(testset, batch_sampler=test_BatchSampler, num_workers = 16, pin_memory = True)
        matrix = torch.zeros([7,7])
        model = model.to(DEVICE)
        start_time = time.time()
        with tqdm(total = testset.length) as bar:
            correct = 0
            start = 0
            bar.set_description('Testing' + str(j))
            total_num = torch.zeros(7).to(DEVICE)
            for i,data in enumerate(testloader):
                image = data[0]
                label = data[1]
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                y_test[j,start : start + image.shape[0]] = label
                total_num += torch.sum(label,dim = 0)
                with torch.no_grad():
                    pre = model(image)
                    if j < len(net_path) - 1:
                        pre = F.softmax(pre,dim = -1)
                    else:
                        pre = torch.sigmoid(pre)
                y_score[j,start : start + image.shape[0]] = pre
                label = anti_onehot(label)
                value,out = pre.max(1,keepdim = True)
                for i in range(label.shape[0]):
                    if value[i] > 0.5:
                        matrix[label[i],out[i]] += 1
                    else:
                        continue
                correct += out.eq(label.view_as(out)).sum().item()
                start += image.shape[0]
                bar.update(label.shape[0])
        end_time = time.time()
        
        duration = end_time - start_time
        print(duration / 2194)
    draw_auc(y_test,y_score)
        

def draw_auc(y_test,y_score,n_classes = 7):
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()

    roc_auc = dict()
    lw=2
    plt.figure()
    
    name = ['Vgg16','Vgg19',
            'ResNeXt-50','ResNet-101','ResNet-50','SE-ResNeXt50','SE-ResNeXt101','SE-ResNet50','SE-ResNet101','InceptionV3','DenseNet121',
            'ours']
    linestyle = ['--',':']
    colors = np.random.rand(len(name),3)
    for j in range(y_test.shape[0]):
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[j, :, i], y_score[j, :, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area（方法二）
            #fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
            #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute macro-average ROC curve and ROC area（方法一）
            
            # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve of {0} (auc = {1:0.3f})'
                    ''.format(name[j],roc_auc["macro"]),
                color=colors[j], linestyle=linestyle[j % len(linestyle)], linewidth=4)
    # Plot all ROC curves莫的了
    '''
    colors = np.random.rand(7,3)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    '''
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("result_new.png",dpi = 600)


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

test1(test_loader,testset)
