"""
  - Blockchain for Federated Learning - 
    Client script 
"""
import os.path

import torch
from fedlearner import *
from blockchain import *
from uuid import uuid4
import requests
from argparse import ArgumentParser
from datasets import *
import time
from typing import Union, Tuple
import torchvision.transforms as transforms


class Client:
    def __init__(self, miner, dataset_dir, updates_dir, name, learning_rate):
        """
        :param miner: 矿工地址
        :param dataset_dir: 数据集存放文件夹
        """
        self.id = str(uuid4()).replace('-', '')
        self.name = name
        self.updates_dir = updates_dir
        self.dataset_dir = dataset_dir
        self.miner = miner
        self.dataset_train, self.dataset_test = self.load_dataset()
        self.learning_rate = learning_rate

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

        block: Union[None, Block] = None
        response = requests.post('http://{node}/block'.format(node=self.miner),
                                 json={'block_head': block_info})
        if response.json()['valid']:
            # 将 urf-8 编码为 bytes，再以 base64 的解码方式解码为二进制 bytes
            block = pickle.loads(codecs.decode(response.json()['block'].encode(), "base64"))
        return block

    def get_model(self, block_info: dict) -> nn.Module:
        """
        根据区块信息获取区块中的模型

        :param block_info: 区块信息
        :return: 模型对象
        """

        block: Block = self.get_full_block(block_info)
        return block.block_body.model_updated

    def get_miner_status(self):
        response = requests.get('http://{node}/status'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()

    def load_dataset(self) -> Tuple[NodeDataset, GlobalDataset]:
        """
        加载数据集

        :return 数据集
        """
        train_transform, test_transform = get_transform()
        dataset_train = NodeDataset(dataset_dir=self.dataset_dir, filename=self.name, transform=train_transform)
        dataset_test = GlobalDataset(dataset_dir=self.dataset_dir, train=False, transform=test_transform)
        return dataset_train, dataset_test

    def update_model(self, model: nn.Module, epochs) -> Tuple[Model, float]:
        """
        client 在本地训练模型

        :param model: 本地当前模型
        :param epochs: 本地训练轮次
        :return:
        """

        t = time.time()
        test_dataloader = DataLoader(self.dataset_test, batch_size=32, shuffle=False, drop_last=False)
        train_dataloader = DataLoader(self.dataset_train, batch_size=64, shuffle=True, drop_last=False)
        worker = NNWorker(train_dataloader=train_dataloader, test_dataloader=test_dataloader, worker_id=self.name,
                          epochs=epochs, device="cuda", learning_rate=self.learning_rate)

        worker.set_model(model)
        worker.train()
        indices = worker.evaluate()
        model_updated = Model(
            model=worker.get_model(),
            accuracy=indices["accuracy"],
            f1_score=indices["f1_score"]
        )
        worker.close()
        return model_updated, time.time()-t

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
                'model_updated': codecs.encode(pickle.dumps(model_updated), "base64").decode(),
                'datasize': len(self.dataset_train),
                'computing_time': cmp_time})
       
    def work(self, epochs, common_rounds) -> None:
        """
        client 本地训练模型并发送交易给区块链

        :param epochs: 训练轮次，每训练完一个轮次将发送至区块链中
        :param common_rounds 共识训练轮次：获取模型，本地训练，发送更新 为一个轮次
        """
        # 区块链中最新区块的区块高度
        base_block_height = -1
        for common_round in range(common_rounds):
            # 等待区块链可接受
            wait = True
            while wait:
                status = client.get_miner_status()
                if status['status'] != "receiving" or base_block_height == status['last_model_index']:
                    time.sleep(10)
                else:
                    wait = False

            # 获取最新区块信息
            latest_block_head = client.get_latest_block()
            base_block_height = latest_block_head['block_height']
            logger.info("区块链中最新区块全局模型的准确率: {}".format(latest_block_head['accuracy']))

            # 开始进行本地训练
            model = client.get_model(latest_block_head)
            model_updated, cmp_time = client.update_model(model, epochs=epochs)

            # 保存梯度更新
            if not os.path.isdir(self.updates_dir):
                os.mkdir(self.updates_dir)
            with open(self.updates_dir + path_separator +
                      "device" + str(self.id) + "_model_v" + str(common_round) + ".block", "wb") as f:
                pickle.dump(model_updated.model, f)
            logger.info("Client节点: {} 本地训练第 {} 次准确率为: {} F1 分数为: {}"
                        .format(self.id, common_round, model_updated.accuracy, model_updated.f1_score))

            client.send_update(model_updated.model, cmp_time, base_block_height)
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--miner', default='127.0.0.1:5000', help='client通过miner节点与区块链进行交互')
    parser.add_argument('-d', '--dataset_dir', default=".\\dataset", help='dataset数据存放文件夹')
    parser.add_argument('-u', '--updates_dir', default=".\\updates", help='updates数据存放文件夹')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='client本地训练的轮次')
    parser.add_argument('-c', '--common_rounds', default=50, type=int, help='client共识训练的轮次')
    parser.add_argument('-n', '--name', default="node_1", type=str, help='client名字')
    parser.add_argument('-l', '--learning_rate', default="0.01", type=float, help='client 本地训练学习率')
    args = parser.parse_args()

    client = Client(miner=args.miner,
                    dataset_dir=args.dataset_dir,
                    name=args.name,
                    updates_dir=args.updates_dir,
                    learning_rate=args.learning_rate)

    logger.info("节点:{} 已完成初始化".format(client.name))

    client.work(args.epochs, args.common_rounds)
