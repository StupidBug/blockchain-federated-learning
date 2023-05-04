import pickle

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import hashlib
from sklearn.metrics import f1_score
import math


def compute_accuracy(model, dataloader, device="cuda"):
    """
    测试模型准确度
    :param model: 模型
    :param dataloader: 数据
    :param device: 设备类型
    :return: 准确率，F1分数
    """

    was_training = False
    # 若现在处于训练模式
    if model.training:
        model.eval()
        was_training = True

    # move the model to cuda device:
    model.to(device)

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data.squeeze()).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
    if was_training:
        model.train()

    accuracy = correct / float(total)
    f1 = f1_score(true_labels_list, pred_labels_list, average='weighted')

    return accuracy, f1


def hash_sha256(text: object):
    """
    sha256 哈希函数
    :param text: 需要哈希的内容
    :return: 哈希后的结果
    """
    return hashlib.sha256(str(pickle.dumps(text)).encode()).hexdigest()


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def get_transform(dataset_type):
    if dataset_type == "cifar10":
        return get_transform_cifar()
    elif dataset_type == "pathmnist":
        return get_transform_mnist()


def get_transform_cifar():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0)
    ])
    # data prep for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(0., 0)
    ])
    return transform_train, transform_test


def get_transform_mnist():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(0., 0)])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(0., 0)])
    return transform_train, transform_test
