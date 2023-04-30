"""
 - Blockchain for Federated Learning -
   Blockchain script 
"""

import json
import pickle
import time
from urllib.parse import urlparse
import requests
import random
import codecs
import log
from torch.utils.data import DataLoader
from federatedlearner import *
from datasets import *
from utils import *
from typing import *
from model import Model
import torchvision.transforms as transforms
import os

logger = log.setup_custom_logger("blockchain")
block_suffix = ".block"
path_separator = '\\'


class Update:
    def __init__(self, client, base_block_height, update: nn.Module, datasize, computing_time, timestamp=time.time()):
        """
        初始化梯度更新

        :param client: 生成该梯度更新的客户端
        :param base_block_height: 该梯度更新基于哪个区块高度的区块中的模型
        :param update: 区块更新
        :param datasize: 数据大小
        :param computing_time: 计算时间
        :param timestamp: 时间戳
        """

        self.timestamp = timestamp
        self.base_block_height = base_block_height
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time


class Block:
    def __init__(self, previous_hash, miner_id, block_height, model_updated: nn.Module, accuracy, updates: list[Update],
                 time_limit, update_limit):
        # 区块体
        self.block_body = self.BlockBody(
            model_updated=model_updated,
            updates=updates)

        # 区块头
        self.block_head = self.BlockHead(
            timestamp=time.time(),
            previous_hash=previous_hash,
            block_height=block_height,
            block_body_hash=hash_sha256(self.block_body),
            miner=miner_id,
            accuracy=accuracy,
            time_limit=time_limit,
            update_limit=update_limit,
            nonce=random.randint(0, 100000000)
        )

    class BlockHead:
        def __init__(self, timestamp, previous_hash, block_height, block_body_hash, miner, accuracy, time_limit,
                     update_limit, nonce):
            self.timestamp = timestamp
            self.previous_hash = previous_hash
            self.block_height = block_height
            self.hash = block_body_hash
            self.miner = miner
            self.accuracy = accuracy
            self.time_limit = time_limit
            self.update_limit = update_limit
            self.nonce = nonce

        def get_block_hash(self):
            """
            获取当前区块的区块哈希

            :return: 区块头的哈希值
            """

            return hash_sha256(self)

    class BlockBody:
        def __init__(self, model_updated, updates):
            self.model_updated = model_updated
            self.updates = updates


class Blockchain(object):
    """
    区块链
    """

    def __init__(self, miner_id, block_dir, dataset_dir, update_limit=10, time_limit=1800):
        """
        初始化区块链, 区块链中只有最新的区块以Block类的形式进行存储，过去的区块
        都是以 list(dict) 的类型进行存储

        :param miner_id: 矿工节点的地址 ip:port
        :param genesis_model: 初始模型
        :param gen: 是否为创世区块
        :param update_limit: 更新次数限制
        :param time_limit: 训练时间限制
        :param block_dir: 区块文件存储位置
        """
        super(Blockchain, self).__init__()
        self.cursor_block: Union[Block, None] = None
        self.miner_id = miner_id
        self.hashchain: list[Block.BlockHead] = []
        self.current_updates: list[Update] = list()
        self.update_limit = update_limit
        self.time_limit = time_limit
        self.block_dir = block_dir
        self.node_addresses = set()
        self.dataset_dir = dataset_dir
        # TODO 矿工的验证集应该怎么设置
        self.dataset_test = self.load_dataset()

    def load_dataset(self) -> GlobalDataset:
        """
        加载数据集

        :return 数据集
        """
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_test = GlobalDataset(root=self.dataset_dir, train=False, transform=transform)
        return dataset_test

    def register_node(self, address):
        """
        注册节点————将节点地址添加到对象的 nodes 集合
        :param address:
        :return:
        """
        if address[:4] != "http":
            address = "http://" + address
        # 解析url通过.netloc获得url中的ip:port部分
        parsed_url = urlparse(address)
        self.node_addresses.add(parsed_url.netloc)
        print("Registered node", address)

    def make_block(self, previous_hash=None, genesis_model: Model = None) -> Block:
        """
        创建区块

        :param previous_hash: 上一区块的哈希值
        :param genesis_model: 创世区块的模型：如果不为空，则使用该模型
        :return: 区块对象
        """

        accuracy = 0
        model_updated = None
        time_limit = self.time_limit
        update_limit = self.update_limit
        if len(self.hashchain) > 0:
            update_limit = self.latest_block.update_limit
            time_limit = self.latest_block.time_limit

        # 当为非创世区块时，使用最新区块的哈希值作为 previous_hash
        if previous_hash is None:
            previous_hash = hash_sha256(self.latest_block)

        # 当该区块为创世区块，则使用给定模型
        if genesis_model is not None:
            accuracy = genesis_model.accuracy
            model_updated = genesis_model.model

        # 存在梯度更新则聚合模型
        elif len(self.current_updates) > 0:
            # TODO
            base = self.cursor_block.block_body.model_updated
            accuracy, model_updated = self.compute_global_model(base, self.current_updates, 1)

        block_height = len(self.hashchain) + 1
        block = Block(
            previous_hash=previous_hash,
            miner_id=self.miner_id,
            block_height=block_height,
            model_updated=model_updated,
            accuracy=accuracy,
            updates=self.current_updates,
            time_limit=time_limit,
            update_limit=update_limit
        )
        return block

    def store_block(self, block: Block) -> None:
        """
        存储区块————完整的区块以文件的形式存储在本地，程序中
        只保存最新区块的完整区块文件，和已共识区块的区块头

        :param block: 区块结构体
        :return:
        """

        # cursor_block 存储区块链中的最新完整区块，只有挖出了下一个区块时，才会将区块存储
        if self.cursor_block is not None:
            if not os.path.isdir(self.block_dir):
                os.mkdir(self.block_dir)
            block_path = self.block_dir + path_separator + "federated_model" + \
                         str(self.cursor_block.block_head.block_height) + block_suffix
            with open(block_path, "wb") as f:
                pickle.dump(self.cursor_block, f)
        self.cursor_block = block

        self.hashchain.append(block.block_head)
        # 清空当前存储的梯度更新
        self.current_updates = list()

    def get_block(self, block_height) -> Union[Block, None]:
        """
        获取指定区块高度的完整区块

        :param block_height: 区块高度
        :return:
        """

        block_path = self.block_dir + path_separator + self.miner_id + path_separator + "/federated_model" + \
                     str(block_height) + block_suffix
        if os.path.isfile(block_path):
            with open(block_path, "rb") as f:
                block = pickle.loads(f.read())
                return block
        return None

    def new_update(self, client, base_block_height, update, datasize, computing_time):
        """
        将接收的梯度更新暂存在current_updates中，等打包区块时，
        再将梯度更新放进去
        """

        self.current_updates.append(Update(
            client=client,
            base_block_height=base_block_height,
            update=update,
            datasize=datasize,
            computing_time=computing_time
        ))
        return self.latest_block.block_height + 1

    @property
    def latest_block(self):
        """
        返回最新区块

        :return: 最新区块
        """

        return self.hashchain[-1]

    def proof_of_work(self, stop_event, genesis_model: Model = None) -> Tuple[Block, bool]:
        """
        工作量证明挖矿

        :param genesis_model: 创世 model
        :param stop_event:
        :return: block_info: 区块信息
        :return: stopped:
        """

        if genesis_model is not None:
            block = self.make_block(previous_hash="-1", genesis_model=genesis_model)
        else:
            block = self.make_block()

        stopped = False
        block_head = block.block_head

        # 挖掘 nonce
        while self.valid_nonce(block_head) is False:
            # 当其他节点挖到了区块，则停止挖矿
            if stop_event.is_set():
                stopped = True
                break
            # nonce 不断增加直至合法
            block_head.nonce += 1
            if block_head.nonce % 1000000 == 0:
                logger.info("mining: {}".format(block_head.nonce))

        # 如果是自己挖到的区块，则存储该区块
        if stopped is False:
            self.store_block(block)
        if stopped:
            logger.info("有其他节点挖掘出了区块")
        else:
            logger.info("区块挖掘成功，区块头为: {}".format(block_head.__dict__))

        return block, stopped

    @staticmethod
    def valid_nonce(block_head: Block.BlockHead):
        """
        验证区块头中的 nonce 是否有效

        :param block_head: 区块头
        :return:
        """

        k = "00000"
        guess_hash = hash_sha256(block_head)
        return guess_hash[:len(k)] == k

    def valid_chain(self, block_chain: list[Block.BlockHead]):
        """
        验证区块链是否合法

        :param block_chain:
        :return:
        """

        last_block_head = block_chain[0]
        current_block_height = 1
        while current_block_height < len(block_chain):
            current_block_head = block_chain[current_block_height]
            # 验证 previous_hash
            if current_block_head.previous_hash != hash_sha256(last_block_head):
                logger.warn("prev_hash diverse", current_block_head)
                return False
            #
            if not self.valid_nonce(current_block_head):
                print("invalid proof", current_block_head)
                return False
            last_block_head = current_block_head
            current_block_head += 1
        return True

    def resolve_conflicts(self, stop_event):
        """
        冲突解决————当其他节点中出现更长且有效的区块链时，将自身维护的区块链替换成最长有效区块链

        :param stop_event:
        :return:
        """

        neighbours = self.node_addresses
        new_chain = None
        bnode = None
        max_length = len(self.hashchain)

        # 获取最长有效链
        for node in neighbours:
            response = requests.get('http://{node}/chain'.format(node=node))
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node

        # 更新节点维护的区块链，并获取最新区块对象
        if new_chain:
            stop_event.set()
            self.hashchain = new_chain
            hblock = self.hashchain[-1]
            rsp = requests.post('http://{node}/block'.format(node=bnode),
                                json={'hblock': hblock})
            self.current_updates = list()
            if rsp.status_code == 200:
                if rsp.json()['valid']:
                    self.cursor_block = pickle.loads(rsp.json()['block'])
            return True
        return False

    def compute_global_model(self, base_model: nn.Module, updates: Union[list[Update], None], learning_rate):

        """
        聚合全局模型
        :param base_model:
        :param updates:
        :param learning_rate:
        :return:
        """

        dataloader_global = DataLoader(self.dataset_test, batch_size=32, shuffle=True)
        worker = NNWorker(train_dataloader=None, test_dataloader=dataloader_global, worker_id="Aggregation",
                          epochs=None, device="cuda", learning_rate=learning_rate)
        worker.build(base_model, updates)
        model = worker.get_model()
        accuracy = worker.evaluate()
        worker.close()
        return accuracy, model
