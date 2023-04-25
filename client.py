"""
  - Blockchain for Federated Learning - 
    Client script 
"""

import torch
from federatedlearner import *
from blockchain import *
from uuid import uuid4
import requests
from argparse import ArgumentParser
from datasets import *
import time


class Client:
    def __init__(self, miner, dataset_dir, client_name):
        """
        :param miner: 矿工地址
        :param dataset_dir: 数据集存放文件夹
        """
        self.id = str(uuid4()).replace('-', '')
        self.name = client_name
        self.dataset_dir = dataset_dir
        self.miner = miner
        self.dataset = self.load_dataset()

    def get_last_block(self):
        """
        获取最新的区块
        :return:
        """
        return self.get_chain()[-1]

    def get_chain(self):
        """
        获取完整的区块链
        :return:
        """
        response = requests.get('http://{node}/chain'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()['chain']

    def get_full_block(self, hblock):
        response = requests.post('http://{node}/block'.format(node=self.miner),
                                 json={'hblock': hblock})
        if response.json()['valid']:
            return Block.from_string(response.json()['block'])
        print("Invalid block!")
        return None

    def get_model(self, hblock):
        response = requests.post('http://{node}/model'.format(node=self.miner),
                                 json={'hblock': hblock})
        if response.json()['valid']:
            return dict(pickle.loads(codecs.decode(response.json()['model'].encode(), "base64")))
        print("Invalid model!")
        return None

    def get_miner_status(self):
        response = requests.get('http://{node}/status'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()

    def load_dataset(self):
        """
        加载数据集
        """

        return NodeDataset(self.dataset_dir, self.name)

    def update_model(self, model, epochs):

        """
        client 在本地训练模型
        :param model: 本地当前模型
        :param epochs: 本地训练轮次
        :return:
        """

        t = time.time()
        dataset = GlobalDataset(self.dataset_dir, train=False)
        dataloader_global = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        # TODO 待确认参数
        dataloader_local = DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=4)
        worker = NNWorker(train_dataloader=dataloader_local, test_dataloader=dataloader_global, worker_id="Aggregation",
                          epochs=epochs, device="cuda")

        worker.build(model)
        worker.train()
        update = worker.get_model()
        accuracy = worker.evaluate()
        worker.close()
        return update, accuracy, time.time()-t

    def send_update(self, update, cmp_time, baseindex):
        """
        向矿工发送更新交易
        :param update:
        :param cmp_time:
        :param baseindex:
        :return:
        """
        logger.info("Client:{} 正在向miner节点:{} 发送梯度更新交易".format(self.id, self.miner))
        requests.post('http://{node}/transactions/new'.format(node=self.miner), json={
                'client': self.id,
                'baseindex': baseindex,
                'update': codecs.encode(pickle.dumps(sorted(update.items())), "base64").decode(),
                # TODO 待更新
                'datasize': len(self.dataset['train_images']),
                'computing_time': cmp_time})
       
    def work(self, epochs):
        """
        client 本地训练模型并发送交易给区块链
        :param epochs: 本地训练轮次
        :return:
        """
        # 最新区块的index
        last_model_index = -1
        for epoch in range(epochs):
            # 等待区块链可接受
            wait = True
            while wait:
                status = client.get_miner_status()
                if status['status'] != "receiving" or last_model_index == status['last_model_index']:
                    time.sleep(10)
                else:
                    wait = False

            # 获取当前区块链信息
            hblock = client.get_last_block()
            base_index = hblock['index']
            last_model_index = base_index
            logger.info("当前区块链中全局模型的准确率: {}".format(hblock['accuracy']))

            # 开始进行本地训练
            model = client.get_model(hblock)
            update, accuracy, cmp_time = client.update_model(model, 10)
            # 保存梯度更新
            with open("clients/device" + str(self.id) + "_model_v" + str(epoch) + ".block", "wb") as f:
                pickle.dump(update, f)
            logger.info("Client节点: {} 本地训练第 {} 次准确率为: {} ".format(self.id, epoch, accuracy))

            client.send_update(update, cmp_time, base_index)
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--miner', default='127.0.0.1:5000', help='client通过miner节点与区块链进行交互')
    parser.add_argument('-d', '--dataset_dir', default='data/mnist.d', help='dataset数据存放文件夹')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='client本地训练的轮次')
    parser.add_argument('-e', '--name', default="node_1", type=str, help='client名字')
    args = parser.parse_args()

    client = Client(args.miner, args.dataset_dir, args.name)
    logger.info("Client节点:{} 已完成初始化".format(client.id))

    client.work(args.epochs)
