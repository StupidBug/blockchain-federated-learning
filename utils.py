import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import hashlib


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cuda"):
    """
    测试模型准确度
    :param model: 模型
    :param dataloader: 数据
    :param get_confusion_matrix: 是否获取混淆矩阵
    :param device: 设备类型
    :return:
    """

    was_training = False
    # 若现在处于训练模式
    if model.training:
        model.eval()
        was_training = True

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
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
        return correct / float(total), conf_matrix

    return correct / float(total)


def hash_sha256(text):
    """
    sha256 哈希函数
    :param text: 需要哈希的内容
    :return: 哈希后的结果
    """
    return hashlib.sha256(text.encode()).hexdigest()
