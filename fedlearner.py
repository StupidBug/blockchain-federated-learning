"""
 - Blockchain for Federated Learning -
      Federated Learning Script
"""

import numpy as np
import pickle
from model import ModelBuilder, ModelIndicator
import log
from utils import *
import torch.optim as optim
import torch.nn as nn

logger = log.setup_custom_logger("FedLearner")


class NNWorker:
    def __init__(self, train_dataloader, test_dataloader, worker_id="nn0", epochs=10, device="cuda",
                 learning_rate=0.01, dataset_type="cifar10"):

        """
        初始化 NN worker 参数
        :param train_dataloader:
        :param test_dataloader:
        :param worker_id:
        :param epochs:
        :param device:
        :param learning_rate:
        """

        self.worker_id = worker_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.dataset_type = dataset_type
        self.model = self.build_base()
        self.epochs = epochs
        self.device = device

    def build(self, base_model, updates):
        """
        基于模型参数和梯度更新 构建新的模型
        """
        if updates is None:
            self.model = base_model
            return

        base_model = self.build_base()
        number_of_updates = len(updates)
        global_para = base_model.cpu().state_dict()
        for index in range(number_of_updates):
            net_para = updates[index].update.cpu().state_dict()

            if index == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * (1 / number_of_updates)
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * (1 / number_of_updates)

        base_model.load_state_dict(global_para)
        self.model = base_model

    def set_model(self, model):
        """
        基于模型参数构建模型
        :param model:
        :return:
        """
        self.model = model

    def build_base(self):
        if self.dataset_type == "cifar10":
            return ModelBuilder.build_cifar10_cnn()
        elif self.dataset_type == "pathmnist":
            return ModelBuilder.build_pathmnist_cnn()

    def train(self):

        """
        训练模型
        :return:
        """
        self.model.to(self.device)
        logger.info("{} 开始本地训练模型".format(self.worker_id))
        test_model_indicator = compute_accuracy(self.model, dataloader=self.test_dataloader, device=self.device)
        logger.info('{} 模型本地训练前测试集指标为: {}'.format(self.worker_id, test_model_indicator.__dict__))
        train_model_indicator = compute_accuracy(model=self.model, dataloader=self.train_dataloader, device=self.device)
        logger.info('{} 模型本地训练前训练集指标为: {}'.format(self.worker_id, train_model_indicator.__dict__))

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate,
                              momentum=0, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().to(self.device)

        cnt = 0
        if type(self.train_dataloader) == type([1]):
            pass
        else:
            self.train_dataloader = [self.train_dataloader]

        for epoch in range(self.epochs):
            epoch_loss_collector = []
            for tmp in self.train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long().squeeze()

                    out = self.model(x)
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    cnt += 1
                    epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('{} Epoch: {} Loss: {}'.format(self.worker_id, epoch, epoch_loss))

        train_model_indicator = compute_accuracy(model=self.model, dataloader=self.train_dataloader, device=self.device)
        test_model_indicator = compute_accuracy(self.model, self.test_dataloader, device=self.device)

        logger.info('{} 模型训练后训练集指标为: {}'.format(self.worker_id, train_model_indicator.__dict__))
        logger.info('{} 模型训练后测试集指标为: {}'.format(self.worker_id, test_model_indicator.__dict__))

        self.model.to('cpu')
        logger.info('{} 模型本地训练结束'.format(self.worker_id))
        return train_model_indicator.accuracy, test_model_indicator.accuracy

    def evaluate(self) -> ModelIndicator:

        """
        使用测试集评估模型准确度
        :return:
        """

        model_indicator = compute_accuracy(model=self.model, dataloader=self.test_dataloader, device="cuda")

        return model_indicator

    def get_model(self):

        return self.model

    def close(self):
        """
        结束模训练过程
        :return:
        """
        # TODO 未完成
        return None
