"""
 - Blockchain for Federated Learning -
      Federated Learning Script
"""

import numpy as np
import pickle
from model import SimpleCNN
import log
from utils import *
import torch.optim as optim
import torch.nn as nn

logger = log.setup_custom_logger("FederatedLearner")


class NNWorker:
    def __init__(self, train_dataloader, test_dataloader, worker_id="nn0", epochs=10, device="cuda",
                 learning_rate=0.01):

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
        self.model = self.build_base()
        self.epochs = epochs
        self.device = device

    def build(self, base_model, updates):

        """
        基于模型参数构建模型
        :return:
        """
        number_of_updates = len(updates)
        global_para = base_model.state_dict()
        for index in range(number_of_updates):
            net_para = updates[index].cpu().state_dict()

            if base_model is None:
                for key in net_para:
                    global_para[key] = net_para[key] * (1 / number_of_updates)
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * (1 / number_of_updates)

        base_model.load_state_dict(global_para)
        self.model = base_model

    @staticmethod
    def build_base():
        return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)

    def train(self):

        """
        训练模型
        :return:
        """

        logger.info('Training network %s' % str(self.worker_id))

        train_acc = compute_accuracy(self.model, self.train_dataloader, device=self.device)
        test_acc, conf_matrix = compute_accuracy(self.model, self.test_dataloader, get_confusion_matrix=True,
                                                 device=self.device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate,
                              momentum=0, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        cnt = 0
        if type(self.train_dataloader) == type([1]):
            pass
        else:
            self.train_dataloader = [self.train_dataloader]

        # writer = SummaryWriter()

        for epoch in range(self.epochs):
            epoch_loss_collector = []
            for tmp in self.train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    out = self.model(x)
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    cnt += 1
                    epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        train_acc = compute_accuracy(self.model, self.train_dataloader, device=self.device)
        test_acc, conf_matrix = compute_accuracy(self.model, self.test_dataloader, get_confusion_matrix=True,
                                                 device=self.device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)

        self.model.to('cpu')
        logger.info(' ** Training complete **')
        return train_acc, test_acc

    def centralized_accuracy(self):

        ''' 
        Function to train the data and calculate centralized accuracy based on 
        evaluating the updated model peformance on test data 
        '''
        cntz_acc = dict()
        cntz_acc['epoch'] = []
        cntz_acc['accuracy'] = []

        self.build_base()
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        for step in range(1, self.num_steps + 1):
            self.sess.run(self.train_op, feed_dict={self.X: self.train_x, self.Y: self.train_y})
            cntz_acc['epoch'].append(step)
            acc = self.evaluate()
            cntz_acc['accuracy'].append(acc)
            print("epoch", step, "accuracy", acc)
        return cntz_acc

    def evaluate(self):

        """
        使用测试集评估模型准确度
        :return:
        """

        return compute_accuracy(model=self.model, dataloader=self.test_dataloader,
                                get_confusion_matrix=False, device="cuda")

    def get_model(self):

        '''
        Function to get the model's trainable_parameter values
        '''
        varsk = {tf.trainable_variables()[i].name[:2]: tf.trainable_variables()[i].eval(self.sess) for i in
                 range(len(tf.trainable_variables()))}
        varsk["size"] = self.size
        return varsk

    def close(self):
        """
        结束模训练过程
        :return:
        """
        # TODO 未完成
        return None
