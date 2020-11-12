import argparse
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sch
import torchvision.transforms as transforms
import os
import sys
import time
import yaml
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from addict import Dict
from sklearn.metrics import f1_score
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start = time.time()
height = 224
width = 224
#データロード
resize_tensor = transforms.Compose([
    transforms.RandomResizedCrop(size=(height, width)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=get_mean(), std=get_std())
])
batch_size = batch_size
num_workers = num_workers
n_classes = n_classes
trainset = datasets.CIFAR10(train=True)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True,  num_workers=num_workers, pin_memory=True)
testset = datasets.CIFAR10(train=False)
testloader = DataLoader(testset, batch_size = batch_size, shuffle = True,  num_workers=num_workers, pin_memory=True)

net_g = models.resnet50().to(device)
# net_h = classifier()
criterion = nn.CrossEntropyLoss()

#モデルロード
net_g.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_g.prm')))

# net_h.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_h.prm')))

#confusion matrix
def print_cmx(y_true, y_pred, title):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (20,16))
    sns.heatmap(df_cmx, annot=True)
    plt.title(title)
    plt.show()

def validate_cmx(dataloader, model_g, model_h, criterion, metric=False):
    losses = AverageMeter("s_loss", ":.4e")
    acc = AverageMeter("s_acc", ":6.2f")

    gts = []
    preds = []

    model_g.eval().to(device)
    model_h.eval().to(device)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data, label = sample
            data, label = data.to(device), label.to(device)
            tmp_batchsize = data.shape[0]
            feat = model_g(data)
            pred = model_h(feat)
            loss = criterion(pred, label)

            acc = accuracy(pred, label, topk=(1,))
            losses.update(loss.item(), tmp_batchsize)
            acc.update(acc[0].item(), tmp_batchsize)

            _, prediction = pred.max(dim=1)
            gts += list(label.to("cpu").numpy())
            preds += list(prediction.to("cpu").numpy())

    print_cmx(gts, preds, "result")

validate_cmx(s_testloader, net_g, net_h, criterion)
    