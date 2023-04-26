"""
 - Blockchain for Federated Learning -
   Blockchain script 
"""

import hashlib
import json
import time
from flask import Flask,jsonify,request
from uuid import uuid4
from urllib.parse import urlparse
import requests
import random
from threading import Thread, Event
import pickle
import codecs
from torch.utils.data import DataLoader
import numpy as np
from federatedlearner import *
from datasets import *


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
                      epochs=None, device="cuda")
    worker.build(base_model, updates)
    model = worker.get_model()
    accuracy = worker.evaluate()
    worker.close()
    return accuracy, model

def find_len(text,strk):

    ''' 
    Function to find the specified string in the text and return its starting position 
    as well as length/last_index
    '''
    return text.find(strk),len(strk)

class Update:
    def __init__(self,client,baseindex,update,datasize,computing_time,timestamp=time.time()):

        ''' 
        Function to initialize the update string parameters
        '''
        self.timestamp = timestamp
        self.baseindex = baseindex
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time

    @staticmethod
    def from_string(metadata):

        ''' 
        Function to get the update string values
        '''
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'baseindex':")
        i3,l3 = find_len(metadata,"'update': ")
        i4,l4 = find_len(metadata,"'client':")
        i5,l5 = find_len(metadata,"'datasize':")
        i6,l6 = find_len(metadata,"'computing_time':")
        baseindex = int(metadata[i2+l2:i3].replace(",",'').replace(" ",""))
        update = dict(pickle.loads(codecs.decode(metadata[i3+l3:i4-1].encode(), "base64")))
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        client = metadata[i4+l4:i5].replace(",",'').replace(" ","")
        datasize = int(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        computing_time = float(metadata[i6+l6:].replace(",",'').replace(" ",""))
        return Update(client, baseindex, update, datasize, computing_time, timestamp)

    def __str__(self):

        ''' 
        Function to return the update string values in the required format
        '''
        return "'timestamp': {timestamp},\
            'baseindex': {baseindex},\
            'update': {update},\
            'client': {client},\
            'datasize': {datasize},\
            'computing_time': {computing_time}".format(
                timestamp=self.timestamp,
                baseindex=self.baseindex,
                update=codecs.encode(pickle.dumps(sorted(self.update.items())), "base64").decode(),
                client=self.client,
                datasize=self.datasize,
                computing_time=self.computing_time
            )


class Block:
    def __init__(self, miner, index, basemodel, accuracy, updates, timestamp=time.time()):

        ''' 
        Function to initialize the update string parameters per created block
        '''
        self.index = index
        self.miner = miner
        self.timestamp = timestamp
        self.basemodel = basemodel
        self.accuracy = accuracy
        self.updates = updates

    @staticmethod
    def from_string(metadata):

        ''' 
        Function to get the update string values per block
        '''
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'basemodel': ")
        i3,l3 = find_len(metadata,"'index':")
        i4,l4 = find_len(metadata,"'miner':")
        i5,l5 = find_len(metadata,"'accuracy':")
        i6,l6 = find_len(metadata,"'updates':")
        i9,l9 = find_len(metadata,"'updates_size':")
        index = int(metadata[i3+l3:i4].replace(",",'').replace(" ",""))
        miner = metadata[i4+l4:i].replace(",",'').replace(" ","")
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        basemodel = dict(pickle.loads(codecs.decode(metadata[i2+l2:i5-1].encode(), "base64")))
        accuracy = float(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        su = metadata[i6+l6:i9]
        su = su[:su.rfind("]")+1]
        updates = dict()
        for x in json.loads(su):
            isep,lsep = find_len(x,"@|!|@")
            updates[x[:isep]] = Update.from_string(x[isep+lsep:])
        updates_size = int(metadata[i9+l9:].replace(",",'').replace(" ",""))
        return Block(miner,index,basemodel,accuracy,updates,timestamp)

    def __str__(self):

        ''' 
        Function to return the update string values in the required format per block
        '''

        return "'index': {index},\
            'miner': {miner},\
            'timestamp': {timestamp},\
            'basemodel': {basemodel},\
            'accuracy': {accuracy},\
            'updates': {updates},\
            'updates_size': {updates_size}".format(
                index = self.index,
                miner = self.miner,
                basemodel = codecs.encode(pickle.dumps(sorted(self.basemodel.items())), "base64").decode(),
                accuracy = self.accuracy,
                timestamp = self.timestamp,
                updates = str([str(x[0])+"@|!|@"+str(x[1]) for x in sorted(self.updates.items())]),
                updates_size = str(len(self.updates))
            )


class Blockchain(object):
    """
    区块链
    """

    def __init__(self, miner_id, base_model=None, gen=False, update_limit=10, time_limit=1800):
        """
        初始化区块链
        :param miner_id: 矿工节点的地址 ip:port
        :param base_model: 初始模型
        :param gen: 是否为创世区块
        :param update_limit: 更新次数限制
        :param time_limit: 训练时间限制
        """
        super(Blockchain, self).__init__()
        self.miner_id = miner_id
        self.curblock = None
        self.hashchain = []
        self.current_updates = dict()
        self.update_limit = update_limit
        self.time_limit = time_limit
        
        if gen:
            genesis, hgenesis = self.make_block(base_model=base_model, previous_hash=1)
            self.store_block(genesis, hgenesis)
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

    def make_block(self, previous_hash=None, base_model=None):
        """
        创建区块
        :param previous_hash: 上一区块的哈希值
        :param base_model: 当前区块基于的模型
        :return: 区块实例， 区块详情字典
        """
        accuracy = 0
        basemodel = None
        time_limit = self.time_limit
        update_limit = self.update_limit
        if len(self.hashchain) > 0:
            update_limit = self.last_block['update_limit']
            time_limit = self.last_block['time_limit']
        if previous_hash is None:
            previous_hash = self.hash(str(sorted(self.last_block.items())))
        if base_model is not None:
            accuracy = base_model['accuracy']
            basemodel = base_model['model']
        elif len(self.current_updates) > 0:
            base = self.curblock.basemodel
            accuracy, basemodel = compute_global_model(base, self.current_updates, 1)
        index = len(self.hashchain)+1
        block = Block(
            miner=self.miner_id,
            index=index,
            basemodel=basemodel,
            accuracy=accuracy,
            updates=self.current_updates
            )
        hash_block = {
            'index': index,
            'hash': self.hash(str(block)),
            'proof': random.randint(0, 100000000),
            'previous_hash': previous_hash,
            'miner': self.miner_id,
            'accuracy': str(accuracy),
            'timestamp': time.time(),
            'time_limit': time_limit,
            'update_limit': update_limit,
            'model_hash': self.hash(codecs.encode(pickle.dumps(sorted(block.basemodel.items())), "base64").decode())
            }
        return block, hash_block

    def store_block(self, block, hashblock):
        """
        存储区块
        :param block: 区块对象
        :param hashblock: 区块的内容
        :return:
        """
        if self.curblock:
            with open("blocks/federated_model" + str(self.curblock.index) + ".block", "wb") as f:
                pickle.dump(self.curblock, f)
        self.curblock = block
        self.hashchain.append(hashblock)
        self.current_updates = dict()
        return hashblock

    def new_update(self, client, baseindex, update, datasize, computing_time):
        self.current_updates[client] = Update(
            client=client,
            baseindex=baseindex,
            update=update,
            datasize=datasize,
            computing_time=computing_time
            )
        return self.last_block['index']+1

    @staticmethod
    def hash(text):
        """
        sha256 哈希函数
        :param text: 需要哈希的内容
        :return: 哈希后的结果
        """
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def last_block(self):
        """
        返回最新区块
        :return: 最新区块
        """
        return self.hashchain[-1]

    def proof_of_work(self, stop_event):
        """
        工作量证明挖矿
        :param stop_event:
        :return:
        """
        block, hblock = self.make_block()
        stopped = False
        while self.valid_proof(str(sorted(hblock.items()))) is False:
            if stop_event.is_set():
                stopped = True
                break
            hblock['proof'] += 1
            if hblock['proof'] % 1000 == 0:
                print("mining", hblock['proof'])
        if stopped is False:
            self.store_block(block, hblock)
        if stopped:
            print("Stopped")
        else:
            print("Done")
        return hblock, stopped

    @staticmethod
    def valid_proof(block_data):
        """
        验证挖的 nonce 是否有效
        :param block_data:
        :return:
        """
        guess_hash = hashlib.sha256(block_data.encode()).hexdigest()
        k = "00000"
        return guess_hash[:len(k)] == k

    def valid_chain(self, hchain):
        last_block = hchain[0]
        curren_index = 1
        while curren_index < len(hchain):
            hblock = hchain[curren_index]
            if hblock['previous_hash'] != self.hash(str(sorted(last_block.items()))):
                print("prev_hash diverso", curren_index)
                return False
            if not self.valid_proof(str(sorted(hblock.items()))):
                print("invalid proof", curren_index)
                return False
            last_block = hblock
            curren_index += 1
        return True

    def resolve_conflicts(self,stop_event):
        neighbours = self.node_addresses
        new_chain = None
        bnode = None
        max_length = len(self.hashchain)
        for node in neighbours:
            response = requests.get('http://{node}/chain'.format(node=node))
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node
        if new_chain:
            stop_event.set()
            self.hashchain = new_chain
            hblock = self.hashchain[-1]
            resp = requests.post('http://{node}/block'.format(node=bnode),
                                 json={'hblock': hblock})
            self.current_updates = dict()
            if resp.status_code == 200:
                if resp.json()['valid']:
                    self.curblock = Block.from_string(resp.json()['block'])
            return True
        return False
