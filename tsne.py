from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from collections import OrderedDict
from addict import Dict
from sklearn.metrics import f1_score
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, Normalize

import numpy as np
import scipy as sp
from sklearn.datasets import fetch_mldata
import sklearn.base
import bhtsne
import matplotlib.pyplot as plot
import matplotlib.cm as cm

# colorlist
colorlist = ["r", "g", "b", "c", "m", "y", "k", "w", "0.3", "0.7"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize_tensor = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=get_mean(), std=get_std())
])

batch_size = 64
num_workers = 2

trainset = datasets.cifar10(train=False, transform=resize_tensor)
trainloader = DataLoader(trainset, batch_size=batchsize)

######## rewrite this section in response to your architecture #############
net_g = resnet50()
net_g.load_state_dict(torch.load("path/to/prm"))
net_g.eval()
net_g.to(device)

feat = []
label = []

with torch.no_grad():
    for data, label in trainloader:
        data, label = data.to(device), label.to(device)
        feat = net_g(data)
        feat.append(feat)
        label.append(label)

feat = sp.array(feat.cpu())
label = sp.array(label.cpu())

print("data processed")


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=100):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed
        self.max_iter = max_iter

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed
        )




bht = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
feat_tsne = bht.fit_transform(feat)

xmin = feat_tsne[:,0].min()
xmax = feat_tsne[:,0].max()
ymin = feat_tsne[:,1].min()
ymax = feat_tsne[:,1].max()

plot.figure( figsize=(16,12) )


for feat, label in zip(feat_tsne, label):
    plot.scatter(feat[0], feat[1], c=colorlist[int(label)], marker=str("$")+str(label)+str("$"))

plot.axis([xmin,xmax,ymin,ymax])
plot.xlabel("component 0")
plot.ylabel("component 1")
plot.title("conventional t-SNE visualization")
plot.savefig("conventional_tsne.png")