import os

import torch

import log

from utils import *
from torch.utils.data import Dataset, random_split
from datasets import NodeDataset, DatasetBuilder
from argparse import ArgumentParser
import torchvision.transforms as transforms

logger = log.setup_custom_logger("data")
path_separator = os.sep
dataset_suffix = '.dataset'


def load_data(filename):
    """
    从二进制文件中读取出数据集
    :param filename: 文件名
    :return: 数据集
    """
    return torch.load(dataset_dir + filename)


def show_dataset_details(dataset, name):
    """
    展示数据集的基本信息
    """
    length = len(dataset)
    logger.info("节点:{} 数据集长度:{}".format(name, length))


def split_dataset(dataset):
    """
    将数据集分割为 n 份
    :param dataset: 数据集
    :return: 分割后的数据集
    """
    subset_size = [len(dataset) // split_count] * split_count
    subset_size[-1] += len(dataset) % split_count
    datasets = random_split(dataset, subset_size)
    return datasets


def get_cifar10_dataset():
    """
    加载 cifar10 数据
    """
    train_ds = DatasetBuilder.build_cifar10(dataset_dir, train=True, transform=None,
                                            target_transform=None, download=True)
    test_ds = DatasetBuilder.build_cifar10(dataset_dir, train=False, transform=None,
                                           target_transform=None, download=True)

    return train_ds, test_ds


def get_medminst_dataset(data_flag):
    """
    加载 medminst 数据
    """
    train_ds = DatasetBuilder.build_medmnist(root=dataset_dir, data_flag=data_flag, train=True,
                                             download=True, transform=None)
    test_ds = DatasetBuilder.build_medmnist(root=dataset_dir, data_flag=data_flag, train=False,
                                            download=True, transform=None)
    return train_ds, test_ds


def save_dataset(dateset, dataset_name):
    """
    保存 dataset
    :param dateset: 数据集
    :param dataset_name: 数据集文件名
    """
    torch.save(dateset, dataset_dir + path_separator + dataset_name + dataset_suffix)


def prepare_data():
    logger.info("开始准备数据集————数据集类型: {} 节点数量为:{} 数据本地存放路径:{}".format(args.dataset_type, split_count, dataset_dir))

    # cifar10 数据集
    if args.dataset_type == "cifar10":
        train_ds, test_ds = get_cifar10_dataset()

    # medmnist 数据集
    elif str(args.dataset_type).__contains__('mnist'):
        train_ds, test_ds = get_medminst_dataset(data_flag=args.dataset_type)

    else:
        logger.error("数据集参数不规范")
        return

    show_dataset_details(train_ds, "全局训练数据集")
    show_dataset_details(test_ds, "全局测试训练集")
    save_dataset(train_ds, "global_train_dataset")
    save_dataset(test_ds, "global_test_dataset")
    for n, d in enumerate(split_dataset(train_ds)):
        node_name = "node_" + str(n)
        save_dataset(d, node_name)
        show_dataset_details(d, node_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', default=".//dataset", help='dataset数据存放文件夹')
    parser.add_argument('-n', '--node_num', default=10, type=int, help='节点数量')
    parser.add_argument('-t', '--dataset_type', default="pathmnist", type=str, help='数据集类型')
    args = parser.parse_args()
    # 数据本地存放路径
    dataset_dir = args.dataset_dir
    # 数据分割数量
    split_count = args.node_num
    # 开始准备数据
    prepare_data()
