import bisect
import copy
import csv
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim import optimizer
from scipy import interp
from sklearn.metrics import auc, roc_curve
from torch.cuda.amp import \
    GradScaler as GradScaler  # only supported in pytorch 1.6
from torch.cuda.amp import \
    autocast as autocast  # only supported in pytorch 1.6
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import BatchSampler
from torchvision.io import read_image
from tqdm import tqdm
from my_summary import summary
from densenet_121 import densenet121
from resnet import resnet50,resnet101,resnext50
from vggnet import vgg16,vgg19
from inception import InceptionV3
from se_resnet import se_resnet50,se_resnet101,se_resnext50,se_resnext101
from densenet import my_densenet
from read_data_pre import train_dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = True

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
            self.batch_imgs = self.batch_imgs.cuda(non_blocking = True)
            self.labels = self.labels.cuda(non_blocking = True)
    
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
        a = relative_order[0]
        b = a.sort()
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

def get_weight(labels,class_num = 7):
    numbers = torch.zeros(class_num)
    for label in labels:
        label = np.argmax(label)
        numbers[label] += 1
    return numbers

def get_set(root,label_dir = 'csv_dir',pre_train = True):
    csv_dir = os.path.join(root,label_dir)
    #csv_dir = '/home/server/Downloads/dataset/medic/label/ISIC2018_Task3_Training_GroundTruth.csv'
    train_set = []
    train_label = []
    val_set = []
    val_label = []
    #root = '/home/server/Downloads/dataset/medic'
    image_dir = 'data_dir'
    means = torch.zeros(3)
    stds = torch.zeros(3)
    if pre_train:
        means = torch.tensor([0.485, 0.456, 0.406])
        stds = torch.tensor([0.229, 0.224, 0.225])
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
                #a = np.array(row[1:],dtype=np.float)
                #print(row[1])
                #file_name = file_name.rstrip('\n')
                #self.image_xml.append(self.train_xml_dir + file_name + '.xml')
                image_label = np.array(row[1:],dtype=np.float)
                image_label = torch.from_numpy(image_label).half()
                image_name = file_name + '.jpg'
                image_name = os.path.join(root,image_dir,image_name)
                np.random.seed(i)
                train_num = np.random.uniform(0,1)
                if train_num < 0.8:   
                    image = read_image(image_name)
                    image.requires_grad = False
                    train_set.append(image_name)
                    train_label.append(image_label)
                    # only available without pretrain
                    if not pre_train:
                        means += torch.mean(image.float(),dim = [1,2])
                        stds += torch.std(image.float(),dim = [1,2])
                    #del(image)
                else:
                    val_set.append(image_name)
                    val_label.append(image_label)
                #self.image.append(image_name)
                #self.image_label.append(image_label)
                bar.update(1)
                i+=1
    numbers = get_weight(train_label)
    if not pre_train:
        means = means.reshape(3,1,1) / len(train_set)
        stds = stds.reshape(3,1,1) / len(train_set)
    return [train_set,train_label],[val_set,val_label],numbers,means,stds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
PRE_TRAIN = True
HAS_TRAINED = 0

model = my_densenet(attention_module = 'CBAM', pre_train = PRE_TRAIN).to(DEVICE)
#model = vgg16().to(DEVICE)
#model = vgg19().to(DEVICE)
#model = resnet50().to(DEVICE)
#model = resnet101().to(DEVICE)
#model = resnext50().to(DEVICE)
#model = InceptionV3().to(DEVICE)
#model = se_resnet50().to(DEVICE)
#model = se_resnet101().to(DEVICE)
#model = se_resnext50().to(DEVICE)
#model = se_resnext101().to(DEVICE)

lr = 1e-4
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params,lr = lr,momentum=0.9,weight_decay=1e-4)
optimizer = optim.Adam(params,lr)
trains,vals,numbers,means,stds = get_set(root ='data_dir', pre_train=PRE_TRAIN)
WARM_UP_STEPS = 100
total_label_number = torch.sum(numbers)
#total_label_number = torch.median(numbers)
scaler = GradScaler() 
weight = total_label_number / (numbers * 7)
weight = weight.to(DEVICE)


trainset = train_dataset(trains[0],trains[1],means,stds)
#train_loader = DataLoader(trainset,batch_size=BATCH_SIZE,num_workers=6,shuffle=True)
train_sampler = make_data_sampler(trainset,True)
train_BatchSampler = make_batch_data_sampler(trainset,train_sampler,1,BATCH_SIZE)
train_loader = DataLoader(trainset, batch_sampler=train_BatchSampler, num_workers = 8, pin_memory = True)
steps = len(train_loader)
train_loader = data_prefetcher(train_loader)
Loss = torch.nn.CrossEntropyLoss()


print('The dataset preparation is done')

def train_step(inputs,epoch):
    #model.to(DEVICE)
    running_loss = 0
    alpha = torch.tensor([0.25,0.25,0.25,0.25,0.25,0.25,0.25]).to(DEVICE)
    min_scale = 1e-4 # changed 1e-2
    max_scale = 0.8
    step = epoch * steps + 1
    #Loss = torch.nn.CrossEntropyLoss()
    with tqdm(total = trainset.length) as bar:
        bar.set_description('EPOCH '+ str(epoch))
        inputs.start()
        data = inputs.next()
        while data[0] != None:
            if step <= WARM_UP_STEPS:
                for param in optimizer.param_groups:
                    param['lr'] = lr * step / WARM_UP_STEPS
            optimizer.zero_grad()
            image,label = data
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            with autocast(): 
                pre = model(image)
                pre = torch.sigmoid(pre)
                pre = pre.clamp(min_scale,max_scale)
                '''
                our focal loss
                '''
                loss = - (alpha * (max_scale - pre)**2 * label * torch.log(pre) + \
                         (1 - alpha) * (pre)**2 * (1 - label) * torch.log((1 - pre))) # (pre - min_scale) 
                loss = torch.sum(loss,dim = 0)
                loss = torch.mean(loss) 
                #if torch.isnan(loss):
                    #print(3)
                #loss = - (alpha * (max_scale - pre)**2 * label * torch.log(pre) + \
                         #(1 - alpha) * (pre)**2 * (1 - label) * torch.log((1 - pre))) 
                #loss = torch.sum(loss,dim = -1)
                #loss = torch.mean(loss)
                #label = anti_onehot(label)
                #loss = Loss(pre,label)
            scaler.scale(loss).backward()  # only supported in pytorch 1.6
            scaler.step(optimizer)  # only supported in pytorch 1.6
            scaler.update()  # only supported in pytorch 1.6
            running_loss += loss.item()
            data = inputs.next()
            bar.update(label.shape[0])
            step += 1
        #bar.set_postfix(loss = running_loss)
    torch.save(model.state_dict(),net_path)
    torch.cuda.empty_cache()
    return running_loss

def train(loader,epoch,has_trained = HAS_TRAINED):
    if has_trained > 0:
        model.load_state_dict(torch.load(net_path))
    model.train()
    loss = 0
    train_loader = loader
    for i in range(has_trained,epoch):
        if i == int(epoch * 0.75) or i == int(epoch * 0.9):
            for param in optimizer.param_groups:
                    param['lr'] = lr * 0.1
        loss = train_step(train_loader,i)
        print("epoch " + str(i) + "th loss is " + str(loss))
        train_sampler = make_data_sampler(trainset,True)
        train_BatchSampler = make_batch_data_sampler(trainset,train_sampler,1,BATCH_SIZE)
        train_loader = DataLoader(trainset, batch_sampler=train_BatchSampler, num_workers = 8, pin_memory = True)
        train_loader = data_prefetcher(train_loader)  
    return loss

def anti_onehot(labels):
    result = torch.zeros(labels.shape[0])
    result = torch.argmax(labels,dim=1)
    return result

def get_recall_precition(matrix,total_num,is_f1_score = True):
    recall = []
    precition = []
    specificity = []
    f1_score = []
    for i in range(matrix.shape[0]):
        temp_class = matrix[i,i]
        #recall_compute =  torch.sum(matrix[i,:])
        precition_compute = torch.sum(matrix[:,i])
        negative = torch.sum(matrix) - torch.sum(matrix[i,:])
        TN = negative - torch.sum(matrix[:,i]) + matrix[i,i]
        specificity.append(float(TN / negative))
        recall.append(float(temp_class / total_num[i]))
        precition.append(float(temp_class / precition_compute))
    f1_score = 2 * np.array(precition) * np.array(recall) / (np.array(precition) + np.array(recall))
    if not is_f1_score:
        return recall, precition, specificity
    else:
        return recall,precition,specificity,f1_score

def test1():
    testset = train_dataset(vals[0],vals[1],means,stds,False)
    #testloader = DataLoader(testset,batch_size=BATCH_SIZE,num_workers=6,shuffle=True)
    test_sampler = make_data_sampler(testset,True)
    test_BatchSampler = make_batch_data_sampler(testset,test_sampler,1,BATCH_SIZE)
    test_loader = DataLoader(testset, batch_sampler=test_BatchSampler, num_workers = 8, pin_memory = True)
    #model = model.to(DEVICE)
    
    matrix = torch.zeros([7,7])
    
    y_test = torch.zeros([testset.length,7])
    y_score = torch.zeros_like(y_test)
    with tqdm(total = testset.length) as bar:
        correct = 0
        start = 0
        bar.set_description('Testing')
        total_num = torch.zeros(7).to(DEVICE)
        for i,data in enumerate(test_loader):
            image = data[0]
            label = data[1]
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            y_test[start : start + image.shape[0]] = label
            total_num += torch.sum(label,dim = 0)
            with torch.no_grad():
                pre = model(image)
                pre = torch.sigmoid(pre)
            y_score[start : start + image.shape[0]] = pre
            label = anti_onehot(label)
            value,out = pre.max(1,keepdim = True)
            for i in range(label.shape[0]):
                #if value[i] > 0.4:
                matrix[label[i],out[i]] += 1
                #else:
                    #continue
            
            correct += out.eq(label.view_as(out)).sum().item()
            start += image.shape[0]
            bar.update(label.shape[0])
    draw_auc(y_test,y_score)
    recall,precition,specifity,f1_score = get_recall_precition(matrix,total_num)
    print('recall = ')
    print(recall)
    print('mean recall =')
    print(np.mean(recall))
    print('precision = ')
    print(precition)
    print('mean precision = ')
    print(np.mean(precition))
    print('sp = ')
    print(specifity)
    print('mean specifity =')
    print(np.mean(specifity))
    print('f1 score')
    print(list(f1_score))
    print('mean f1 score')
    print(np.mean(f1_score))
    print('accuracy')
    print(correct / len(test_loader.dataset))


def draw_auc(y_test,y_score,n_classes = 7):
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
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

    # Plot all ROC curves莫的了
    lw=2
    plt.figure()
    '''
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    '''
    name = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.3f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    colors = np.random.rand(7,7)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.3f})'
                ''.format(name[i], roc_auc[i]))
 
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(net_path + '.jpg',dpi = 600)

net_path = 'net_path'
net = train(train_loader,100)
model.load_state_dict(torch.load(net_path))
model.eval()
test1()




