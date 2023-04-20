import pickle
import torchvision.transforms as transforms
import log
from datasets import Cifar10Truncated


def save_data(datasets, name="cifar10.d"):
    """
    将数据集以二进制形式存储起来
    :param datasets: 数据集
    :param name: 文件名
    """
    with open(name, "wb") as f:
        pickle.dump(datasets, f)


def load_data(name="cifar10.d"):
    """
    从二进制文件中读取出数据集
    :param name: 文件名
    :return: 数据集
    """
    with open(name, "rb") as f:
        return pickle.load(f)


def show_dataset_details(dataset, name):
    """
    展示数据集的基本信息
    :param dataset: 数据集
    """
    for k in dataset.keys():
        logger.info("%s: %s: %s", name, k, dataset[k].shape)


def split_dataset(dataset, split_count):
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
    save_data(get_cifar10_dataset())
    cifar10_dataset = load_data()
    show_dataset_details(cifar10_dataset, "cifar10_global")
    for n, d in enumerate(split_dataset(cifar10_dataset, 2)):
        node_data_name = "federated_data_" + str(n) + ".d"
        save_data(d, node_data_name)
        dataset = load_data(node_data_name)
        show_dataset_details(dataset, node_data_name)


if __name__ == '__main__':
    logger = log.setup_custom_logger("data")
    dataset_dir_path = "/tmp/dataset"
    prepare_data()
