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
from typing import Tuple

logger = log.setup_custom_logger("blockchain")


def compute_global_model(base_model, updates, learning_rate):

    """
    聚合全局模型
    :param base_model:
    :param updates:
    :param learning_rate:
    :return:
    """

    dataset = GlobalDataset("d:/dataset", train=False)
    dataloader_global = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    worker = NNWorker(train_dataloader=None, test_dataloader=dataloader_global, worker_id="Aggregation",
                      epochs=None, device="cuda", learning_rate=learning_rate)
    worker.build(base_model, updates)
    model = worker.get_model()
    accuracy = worker.evaluate()
    worker.close()
    return accuracy, model


def find_len(text, strk):

    ''' 
    Function to find the specified string in the text and return its starting position 
    as well as length/last_index
    '''
    return text.find(strk), len(strk)


class Update:
    def __init__(self, client, baseindex, update, datasize, computing_time, timestamp=time.time()):

        ''' 
        Function to initialize the update string parameters
        '''
        self.timestamp = timestamp
        self.baseindex = baseindex
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time


class Block:
    def __init__(self, previous_hash, miner_id, block_height, base_model: nn.Module, accuracy, updates: dict[Update],
                 time_limit, update_limit):

        # 区块体
        self.block_body = self.BlockBody(
            base_model=base_model,
            updates=updates)

        # 区块头
        self.block_head = self.BlockHead(
            timestamp=time.time(),
            previous_hash=previous_hash,
            block_height=block_height,
            merkle_root=hash_sha256(self.block_body),
            miner=miner_id,
            accuracy=accuracy,
            time_limit=time_limit,
            update_limit=update_limit,
            nonce=random.randint(0, 100000000)
        )

    class BlockHead:
        def __init__(self, timestamp, previous_hash, block_height, merkle_root, miner, accuracy, time_limit,
                     update_limit, nonce):
            self.timestamp = timestamp
            self.previous_hash = previous_hash
            self.block_height = block_height
            self.merkle_root = merkle_root
            self.miner = miner
            self.accuracy = accuracy
            self.time_limit = time_limit
            self.update_limit = update_limit
            self.nonce = nonce

    class BlockBody:
        def __init__(self, base_model, updates):
            self.base_model = base_model
            self.updates = updates


class Blockchain(object):
    """
    区块链
    """

    def __init__(self, miner_id, base_model=None, gen=False, update_limit=10, time_limit=1800):
        """
        初始化区块链, 区块链中只有最新的区块以Block类的形式进行存储，过去的区块
        都是以 list(dict) 的类型进行存储

        :param miner_id: 矿工节点的地址 ip:port
        :param base_model: 初始模型
        :param gen: 是否为创世区块
        :param update_limit: 更新次数限制
        :param time_limit: 训练时间限制
        """
        super(Blockchain, self).__init__()
        self.cursor_block = None
        self.miner_id = miner_id
        self.hashchain: list[Block.BlockHead] = []
        self.current_updates = dict()
        self.update_limit = update_limit
        self.time_limit = time_limit

        # 构造创世区块，并添加入区块链
        if gen:
            genesis_block = self.make_block(base_model=base_model, previous_hash=1)
            self.store_block(genesis_block)

        self.node_addresses = set()

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

    def make_block(self, previous_hash=None, base_model=None) -> Block:
        """
        创建区块

        :param previous_hash: 上一区块的哈希值
        :param base_model: 当前区块基于的模型
        :return: 区块对象
        """

        accuracy = 0
        model = None
        time_limit = self.time_limit
        update_limit = self.update_limit
        if len(self.hashchain) > 0:
            update_limit = self.latest_block.update_limit
            time_limit = self.latest_block.time_limit
        if previous_hash is None:
            previous_hash = hash_sha256(str(self.latest_block))
        if base_model is not None:
            accuracy = base_model['accuracy']
            model = base_model['model']
        elif len(self.current_updates) > 0:
            base = self.cursor_block.basemodel
            accuracy, basemodel = compute_global_model(base, self.current_updates, 1)
        block_height = len(self.hashchain) + 1
        block = Block(
            previous_hash=previous_hash,
            miner_id=self.miner_id,
            block_height=block_height,
            base_model=model,
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

        if self.cursor_block is not None:
            with open("blocks/federated_model" + str(self.cursor_block.block_height) + ".block", "wb") as f:
                pickle.dump(self.cursor_block, f)
            self.cursor_block = block

        self.hashchain.append(block.block_head)
        # 清空当前存储的梯度更新
        self.current_updates = dict()

    def new_update(self, client, baseindex, update, datasize, computing_time):
        self.current_updates[client] = Update(
            client=client,
            baseindex=baseindex,
            update=update,
            datasize=datasize,
            computing_time=computing_time
            )
        return self.latest_block.block_height + 1

    @property
    def latest_block(self):
        """
        返回最新区块

        :return: 最新区块
        """

        return self.hashchain[-1]

    def proof_of_work(self, stop_event) -> Tuple[Block, bool]:
        """
        工作量证明挖矿

        :param stop_event:
        :return: block_info: 区块信息
        :return: stopped:
        """

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
            if block_head.nonce % 1000 == 0:
                logger.info("mining: {}".format(block_head.nonce))

        # 如果是自己挖到的区块，则存储该区块
        if stopped is False:
            self.store_block(block)
        if stopped:
            logger.info("有其他节点挖掘出了区块")
        else:
            logger.info("区块挖掘结束")

        return block, stopped

    @staticmethod
    def valid_nonce(block_head: Block.BlockHead):
        """
        验证区块头中的 nonce 是否有效

        :param block_head: 区块头
        :return:
        """

        k = "00000"
        guess_hash = hash_sha256(str(block_head).encode()).hexdigest()
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
            if current_block_head.previous_hash != hash_sha256(str(last_block_head)):
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
            self.current_updates = dict()
            if rsp.status_code == 200:
                if rsp.json()['valid']:
                    self.cursor_block = pickle.loads(rsp.json()['block'])
            return True
        return False
