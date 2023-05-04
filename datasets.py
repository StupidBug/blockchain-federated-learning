import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data.dataset import T_co
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
import log

logger = log.setup_custom_logger("dataset")
dataset_suffix = ".dataset"

path_separator = '\\'


class GlobalDataset(data.Dataset):

    def __init__(self, dataset_dir, train=True, transform=None, target_transform=None):
        self.dataset_dir = dataset_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = self.load_dataset()

    def load_dataset(self):
        if self.train:
            filename = "global_train_dataset"
        else:
            filename = "global_test_dataset"
        dataset = torch.load(self.dataset_dir + path_separator + filename + dataset_suffix)
        logger.info("全局数据集文件: {} 加载完成".format(filename))
        return dataset

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


class NodeDataset(data.Dataset):

    def __init__(self, dataset_dir, filename, transform=None):
        self.dataset_dir = dataset_dir
        self.filename = filename
        self.transform = transform
        self.dataset = self.load_dataset()

    def load_dataset(self):
        dataset = torch.load(self.dataset_dir + path_separator + self.filename + dataset_suffix)
        return dataset

    def __getitem__(self, index) -> T_co:
        img, target = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


class DatasetBuilder:
    @staticmethod
    def build_medmnist(root, data_flag='pathmnist', train=True, download=True, transform=None):
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        # load the data
        if train:
            dataset = DataClass(root=root, split='train', transform=transform, download=download)
        else:
            dataset = DataClass(root=root, split='test', transform=transform, download=download)
        return dataset

    @staticmethod
    def build_cifar10(root, train=True, transform=None, target_transform=None, download=True):
        cifar10 = CIFAR10(root, train, transform, target_transform, download)
        return cifar10
