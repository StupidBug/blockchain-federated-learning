import pickle
import torchvision.transforms as transforms
import log
from datasets import Cifar10Truncated


logger = log.setup_custom_logger("data")


def save_data(dataset, dataset_path):
    """
    将数据集以二进制形式存储起来
    :param dataset_path: 数据集路径
    :param dataset: 数据集
    """
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)


def load_data(dataset_path):
    """
    从二进制文件中读取出数据集
    :param dataset_path: 文件名
    :return: 数据集
    """
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def show_dataset_details(dataset, dataset_name):
    """
    展示数据集的基本信息
    :param dataset_name: 文件
    :param dataset: 数据集
    """
    for k in dataset.keys():
        logger.info("%s: %s: %s", dataset_name, k, dataset[k].shape)


def split_dataset(dataset):
    """
    将数据集分割为 n 份
    :param dataset: 数据集
    :param split_count: 分割份数
    :return: 分割后的数据集
    """
    datasets = []
    split_data_length = len(dataset["train_images"]) // split_count
    for i in range(split_count):
        datasets.append({
            "train_images": dataset["train_images"][i * split_data_length:(i + 1) * split_data_length],
            "train_labels": dataset["train_labels"][i * split_data_length:(i + 1) * split_data_length],
            "test_images": dataset["test_images"][:],
            "test_labels": dataset["test_labels"][:],
        })
    return datasets


def get_cifar10_dataset():
    """
    加载 cifar10 数据
    """

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = Cifar10Truncated(dataset_dir_path, train=True, download=True, transform=transform)
    cifar10_test_ds = Cifar10Truncated(dataset_dir_path, train=False, download=True, transform=transform)

    x_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    x_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    dataset = {
        "train_images": x_train,
        "train_labels": y_train,
        "test_images": x_test,
        "test_labels": y_test,
    }

    return dataset


def prepare_data():
    dataset_path = dataset_dir_path + "cifar10.d"
    save_data(get_cifar10_dataset(), dataset_path)
    cifar10_dataset = load_data(dataset_path)
    show_dataset_details(cifar10_dataset, "cifar10")
    for n, d in enumerate(split_dataset(cifar10_dataset, 2)):
        node_data_name = "federated_data_" + str(n) + ".d"
        node_data_path = dataset_dir_path + node_data_name
        save_data(d, node_data_path)
        dataset = load_data(node_data_path)
        show_dataset_details(dataset, node_data_name)


if __name__ == '__main__':
    # 数据本地存放路径
    dataset_dir_path = "/tmp/dataset/"
    # 数据分割数量
    split_count = 2
    # 开始准备数据
    prepare_data()
