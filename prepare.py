import log

from utils import *
from torch.utils.data import Dataset, random_split
from datasets import GlobalDataset, NodeDataset
import torchvision.transforms as transforms

logger = log.setup_custom_logger("data")


def save_data(dataset, filename):
    """
    将数据集存储
    :param filename: 数据集路径
    :param dataset: 数据集
    """
    torch.save(dataset, dataset_dir_path + filename)


def load_data(filename):
    """
    从二进制文件中读取出数据集
    :param filename: 文件名
    :return: 数据集
    """
    return torch.load(dataset_dir_path + filename)


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
    datasets = random_split(dataset.data, subset_size)
    return datasets


def get_cifar10_dataset():
    """
    加载 cifar10 数据
    """

    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = GlobalDataset(dataset_dir_path, train=True, transform=transform, name="CIFAR10_TRAIN")
    test_ds = GlobalDataset(dataset_dir_path, train=False, transform=transform, name="CIFAR10_TEST")

    return train_ds, test_ds


def prepare_data():
    logger.info("开始数据准备工作————数据本地存放路径:{}".format(dataset_dir_path))
    train_ds, test_ds = get_cifar10_dataset()
    show_dataset_details(train_ds)
    show_dataset_details(test_ds)
    for n, d in enumerate(split_dataset(train_ds)):
        node_name = "node_" + str(n)
        node_dataset = NodeDataset(dataset_dir=dataset_dir_path, name=node_name, dataset=d)
        show_dataset_details(node_dataset)


if __name__ == '__main__':
    # 数据本地存放路径
    dataset_dir_path = "/tmp/dataset/"
    # 数据分割数量
    split_count = 5
    # 开始准备数据
    prepare_data()
