import torch

import log

from utils import *
from torch.utils.data import Dataset, random_split
from datasets import GlobalDataset, NodeDataset
from argparse import ArgumentParser
import torchvision.transforms as transforms

logger = log.setup_custom_logger("data")
path_separator = '\\'
dataset_suffix = '.dataset'


def save_data(dataset, filename):
    """
    将数据集存储
    :param filename: 数据集路径
    :param dataset: 数据集
    """
    torch.save(dataset, dataset_dir + filename)


def load_data(filename):
    """
    从二进制文件中读取出数据集
    :param filename: 文件名
    :return: 数据集
    """
    return torch.load(dataset_dir + filename)


def show_dataset_details(dataset):
    """
    展示数据集的基本信息
    """
    node_name = dataset.name
    length = len(dataset)
    logger.info("节点:{} 数据集长度:{}".format(node_name, length))


def split_dataset(dataset):
    """
    将数据集分割为 n 份
    :param dataset: 数据集
    :return: 分割后的数据集
    """
    subset_size = [len(dataset) // split_count] * split_count
    datasets = random_split(dataset, subset_size)
    return datasets


def get_cifar10_dataset():
    """
    加载 cifar10 数据
    """

    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = GlobalDataset(dataset_dir, train=True, transform=transform, name="CIFAR10_TRAIN")
    test_ds = GlobalDataset(dataset_dir, train=False, transform=transform, name="CIFAR10_TEST")

    return train_ds, test_ds


def save_dataset(dateset, dataset_name):
    """
    保存 dataset
    :param dateset: 数据集
    :param dataset_name: 数据集文件名
    """
    torch.save(dateset, dataset_dir + path_separator + dataset_name + dataset_suffix)


def prepare_data():
    train_ds, test_ds = get_cifar10_dataset()
    show_dataset_details(train_ds)
    save_dataset(train_ds, train_ds.name)
    save_dataset(test_ds, test_ds.name)
    show_dataset_details(test_ds)
    for n, d in enumerate(split_dataset(train_ds)):
        node_name = "node_" + str(n)
        node_dataset = NodeDataset(dataset_dir=dataset_dir, name=node_name, dataset=d)
        save_dataset(node_dataset, node_dataset.name)
        show_dataset_details(node_dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', default=".\\dataset", help='dataset数据存放文件夹')
    parser.add_argument('-n', '--node_num', default=5, type=int, help='节点数量')
    args = parser.parse_args()
    # 数据本地存放路径
    dataset_dir = args.dataset_dir
    # 数据分割数量
    split_count = args.node_num
    logger.info("开始准备数据集————节点数量为:{} 数据本地存放路径:{}".format(split_count, dataset_dir))
    prepare_data()
