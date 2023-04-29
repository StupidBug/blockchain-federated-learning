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
from typing import Union, Tuple


class Client:
    def __init__(self, miner, dataset_dir, name):
        """
        :param miner: 矿工地址
        :param dataset_dir: 数据集存放文件夹
        """
        self.id = str(uuid4()).replace('-', '')
        self.name = name
        self.dataset_dir = dataset_dir
        self.miner = miner
        self.dataset = self.load_dataset()

    def get_latest_block(self) -> dict:
        """
        获取最新的区块

        :return:
        """
        return self.get_chain()[-1]

    def get_chain(self) -> list[dict]:
        """
        获取完整的区块链

        :return: 完整的区块信息列表
        """
        response = requests.get('http://{node}/chain'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()['chain']

    def get_full_block(self, block_info: dict) -> Union[Block, None]:
        """
        根据区块信息获取完整的区块对象

        :param block_info: 区块信息
        :return: 区块对象
        """

        block = None
        response = requests.post('http://{node}/block'.format(node=self.miner),
                                 json={'block_info': block_info})
        if response.json()['valid']:
            block = pickle.loads(response.json()['block'])
        return block

    def get_model(self, block_info: dict) -> nn.Module:
        """
        根据区块信息获取区块中的模型

        :param block_info: 区块信息
        :return: 模型对象
        """

        block: Block = self.get_full_block(block_info)
        return block.block_body.base_model

    def get_miner_status(self):
        response = requests.get('http://{node}/status'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()

    def load_dataset(self) -> data.Dataset:
        """
        加载数据集

        :return 数据集
        """

        return NodeDataset(self.dataset_dir, self.name)

    def update_model(self, model, epochs) -> Tuple[nn.Module, float, float]:
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
        model_updated = worker.get_model()
        accuracy = worker.evaluate()
        worker.close()
        return model_updated, accuracy, time.time()-t

    def send_update(self, model_updated: nn.Module, cmp_time, base_block_height):
        """
        向矿工发送更新交易

        :param model_updated:
        :param cmp_time:
        :param base_block_height:
        :return:
        """
        logger.info("Client:{} 正在向miner节点:{} 发送梯度更新交易".format(self.id, self.miner))
        requests.post('http://{node}/transactions/new'.format(node=self.miner), json={
                'client': self.id,
                'base_block_height': base_block_height,
                # TODO 学习这个编码
                'update': codecs.encode(pickle.dumps(model_updated), "base64").decode(),
                'datasize': len(self.dataset['train_images']),
                'computing_time': cmp_time})
       
    def work(self, epochs) -> None:
        """
        client 本地训练模型并发送交易给区块链

        :param epochs: 训练轮次，每训练完一个轮次将发送至区块链中
        """
        # 区块链中最新区块的区块高度
        latest_block_height = -1
        for epoch in range(epochs):
            # 等待区块链可接受
            wait = True
            while wait:
                status = client.get_miner_status()
                if status['status'] != "receiving" or latest_block_height == status['last_model_index']:
                    time.sleep(10)
                else:
                    wait = False

            # 获取最新区块信息
            latest_block = client.get_latest_block()
            base_block_height = latest_block['block_height']
            logger.info("区块链中最新区块全局模型的准确率: {}".format(latest_block['accuracy']))

            # 开始进行本地训练
            model = client.get_model(latest_block)
            model_updated, accuracy, cmp_time = client.update_model(model, 10)
            # 保存梯度更新
            with open("clients/device" + str(self.id) + "_model_v" + str(epoch) + ".block", "wb") as f:
                pickle.dump(model_updated, f)
            logger.info("Client节点: {} 本地训练第 {} 次准确率为: {} ".format(self.id, epoch, accuracy))

            client.send_update(model_updated, cmp_time, base_block_height)
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--miner', default='127.0.0.1:5000', help='client通过miner节点与区块链进行交互')
    parser.add_argument('-d', '--dataset_dir', default=".\\dataset", help='dataset数据存放文件夹')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='client本地训练的轮次')
    parser.add_argument('-n', '--name', default="node_1", type=str, help='client名字')
    args = parser.parse_args()

    client = Client(miner=args.miner,
                    dataset_dir=args.dataset_dir,
                    name=args.name)

    logger.info("Client节点:{} 已完成初始化".format(client.name))

    client.work(args.epochs)
