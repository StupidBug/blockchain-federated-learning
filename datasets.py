import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data.dataset import T_co
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset
import log

logger = log.setup_custom_logger("dataset")
dataset_suffix = ".dataset"

path_separator = '\\'

class GlobalDataset(data.Dataset):
    """
    cifar10数据集，继承 data.Dataset 类，可被用于生成 dataloader 进行训练
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, name=None):
        """
        构造 cifar10 数据集，如果没有就下载
        """

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.name = name
        # data 变量名固定，不能更改
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        # 如果不存在就会下载数据集
        cifar10 = CIFAR10(self.root, self.train, self.transform, self.target_transform, True)

        dataset = cifar10.data
        target = np.array(cifar10.targets)

        return dataset, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class NodeDataset(data.Dataset):

    def __init__(self, dataset_dir, name, dataset=None):
        self.name = name
        self.dataset_dir = dataset_dir
        self.data = self.load_dataset(dataset)

    def load_dataset(self, dataset):
        if dataset is None:
            return self.load_dataset_from_local()
        else:
            return dataset

    def load_dataset_from_local(self):
        try:
            dataset = torch.load(self.dataset_dir + path_separator + self.name + dataset_suffix)
        except Exception:
            logger.error("节点:{} 无法从本地读取节点数据集", self.name)
            raise FileNotFoundError
        logger.info("节点:{} 数据集已加载成功".format(self.name))
        return dataset

    def __getitem__(self, index) -> T_co:
        return self.data.__getitem__(index)

    def __len__(self):
        return len(self.data)
