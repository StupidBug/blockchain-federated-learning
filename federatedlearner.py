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
        self.net = self.build_base()
        self.epochs = epochs
        self.device = device

    def build(self, base):

        ''' 
        Function to initialize/build network based on updated values received 
        from blockchain
        '''

        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.weights = {
            'w1': tf.Variable(base['w1'], name="w1"),
            'w2': tf.Variable(base['w2'], name="w2"),
            'wo': tf.Variable(base['wo'], name="wo")
        }
        self.biases = {
            'b1': tf.Variable(base['b1'], name="b1"),
            'b2': tf.Variable(base['b2'], name="b2"),
            'bo': tf.Variable(base['bo'], name="bo")
        }

        self.layer_1 = tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1'])
        self.layer_2 = tf.add(tf.matmul(self.layer_1, self.weights['w2']), self.biases['b2'])
        self.out_layer = tf.matmul(self.layer_2, self.weights['wo']) + self.biases['bo']
        self.logits = self.out_layer
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    @staticmethod
    def build_base():
        return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)

    def train(self):

        """
        训练模型
        :return:
        """

        logger.info('Training network %s' % str(self.worker_id))

        train_acc = compute_accuracy(self.net, self.train_dataloader, device=self.device)
        test_acc, conf_matrix = compute_accuracy(self.net, self.test_dataloader, get_confusion_matrix=True,
                                                 device=self.device)

        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate,
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

                    out = self.net(x)
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    cnt += 1
                    epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        train_acc = compute_accuracy(self.net, self.train_dataloader, device=self.device)
        test_acc, conf_matrix = compute_accuracy(self.net, self.test_dataloader, get_confusion_matrix=True,
                                                 device=self.device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)

        self.net.to('cpu')
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

        '''
        Function to calculate accuracy on test data
        '''
        return self.sess.run(self.accuracy, feed_dict={self.X: self.test_x, self.Y: self.test_y})

    def get_model(self):

        '''
        Function to get the model's trainable_parameter values
        '''
        varsk = {tf.trainable_variables()[i].name[:2]: tf.trainable_variables()[i].eval(self.sess) for i in
                 range(len(tf.trainable_variables()))}
        varsk["size"] = self.size
        return varsk

    def close(self):

        '''
        Function to close the current session
        '''
        self.sess.close()
