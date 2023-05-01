"""
 - Blockchain for Federated Learning -
           Mining script
"""
import json
import pickle

from blockchain import *
from threading import Thread, Event
from fedlearner import *
from datasets import GlobalDataset, NodeDataset
from flask import *
from uuid import uuid4
import torchvision.transforms as transforms
from model import SimpleCNN, Model
import codecs
import os
import glob
import log

"""设置全局日志"""
logger = log.setup_custom_logger("miner")
path_separator = "\\"


def make_base(dataset_dir):
    """
    在初始化区块链的时候，需要通过 make_base 创建一个初始模型，然后将该模型添加区块链中
    随后 client 提交的 updates 将在这个初始模型上进行更改
    :return:
    """
    # TODO 是否使用多种模型（看进度）
    # 必加，原理暂不清楚
    transform = transforms.Compose([transforms.ToTensor()])
    global_dataset = GlobalDataset(root=dataset_dir, train=False, transform=transform)
    dataloader_global = DataLoader(global_dataset, batch_size=32, shuffle=True)
    worker = NNWorker(train_dataloader=None, test_dataloader=dataloader_global, worker_id="Aggregation",
                      epochs=None, device="cuda")
    worker.build_base()

    base_model = Model(
        model=worker.get_model(),
        accuracy=worker.evaluate()
    )
    worker.close()
    return base_model


class PoWThread(Thread):
    """
    工作量证明挖矿
    """
    def __init__(self, stop_event, blockchain, node_identifier, genesis_model: Model = None):
        self.stop_event = stop_event
        Thread.__init__(self)
        self.blockchain = blockchain
        self.node_identifier = node_identifier
        self.response = None
        self.genesis_model = genesis_model

    def run(self):
        block, stopped = self.blockchain.proof_of_work(self.stop_event, self.genesis_model)
        self.response = {
            'message': "End mining",
            'stopped': stopped,
            'block': str(block)
        }
        on_end_mining(stopped)


STOP_EVENT = Event()

app = Flask(__name__)

"""
区块链中的每个节点维护一个 status, 包含作为区块链节点所需的全部信息
"""
status = {
    's': "receiving",                           # 节点是否可以接受交易
    'id': str(uuid4()).replace('-', ''),        # 节点的唯一ID
    'blockchain': None,                         # 节点维护的区块链
    'address': ""                               # 节点的IP和Port
    }


def mine(genesis_model=None):
    """
    挖矿

    :param genesis_model: 挖掘创世区块，需要提供创世模型
    :return:
    """

    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT, status["blockchain"], status["id"], genesis_model=genesis_model)
    status['s'] = "mining"
    thread.start()


def on_end_mining(stopped):
    """
    通知其他节点停止挖取区块

    :param stopped:
    :return:
    """
    if status['s'] == "receiving":
        return
    if stopped:
        status["blockchain"].resolve_conflicts(STOP_EVENT)
    status['s'] = "receiving"
    for node in status["blockchain"].node_addresses:
        requests.get('http://{node}/stopmining'.format(node=node))


@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    """
    处理新的一笔交易

    :return:
    """

    logger.info("接收到更新交易")
    # 区块链状态校验
    if status['s'] != "receiving":
        logger.warning("Miner 当前状态不接收交易")
        return 'Miner not receiving', 400

    # 参数合法性校验
    values = request.get_json()
    required = ['client', 'base_block_height', 'model_updated', 'datasize', 'computing_time']
    if not all(k in values for k in required):
        logger.warn("请求参数缺少必须值")
        return 'Missing values', 400
    if values['client'] in status['blockchain'].current_updates:
        logger.warn("请求参数的更新已经被存储")
        return 'Model already stored', 400

    index = status['blockchain'].new_update(values['client'],
                                            values['base_block_height'],
                                            pickle.loads(codecs.decode(values['model_updated'].encode(), "base64")),
                                            values['datasize'],
                                            values['computing_time'])
    # 向所有miner节点转发该交易
    for node in status["blockchain"].node_addresses:
        requests.post('http://{node}/transactions/new'.format(node=node), json=request.get_json())

    # 交易合法性校验，成功则开始 mine
    if (status['s'] == 'receiving' and (
        len(status["blockchain"].current_updates) >= status['blockchain'].latest_block.update_limit or
            time.time()-status['blockchain'].latest_block.timestamp > status['blockchain'].latest_block.time_limit)):
        mine()
    response = {'message': "Update will be added to block {index}".format(index=index)}
    logger.info("该更新请求将被添加至区块中")
    return jsonify(response), 201


@app.route('/status', methods=['GET'])
def get_status():
    """
    获取 miner 状态

    :return 节点状态: [receiving | mining]
    """

    response = {
        'status': status['s'],
        'last_model_index': status['blockchain'].latest_block.block_height
        }
    return jsonify(response), 200


@app.route('/chain', methods=['GET'])
def full_chain():
    """
    获取完整的区块链

    :return: 区块信息列表（区块链）
    """
    hashchain = []
    for block_head in status['blockchain'].hashchain:
        hashchain.append(block_head.__dict__)

    response = {
        'chain': hashchain,
        'length': len(status['blockchain'].hashchain)
    }
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    """
    注册节点————将节点添加到本地的区块链中，并向其他区块链节点发送注册请求

    :return: 已注册矿工节点列表
    """

    values = request.get_json()
    node_addresses = values.get('nodes')

    if node_addresses is None:
        return "Error: Enter valid nodes in the list ", 400

    for node_address in node_addresses:
        if node_address != status['address'] and node_address not in status['blockchain'].node_addresses:
            # 向区块链中添加该节点
            status['blockchain'].register_node(node_address)
            for node_address_registered in status['blockchain'].node_addresses:
                # 向区块链中的其他节点注册该节点
                if node_address_registered != node_address:
                    logger.info("向节点: {} 发送请求注册节点: {}".format(node_address_registered, node_address))
                    requests.post('http://{node_address}/nodes/register'.format(node_address=node_address_registered),
                                  json={'nodes': [node_address]})

    response = {
        'message': "New nodes have been added",
        'total_nodes': list(status['blockchain'].node_addresses)
    }
    return jsonify(response), 201


@app.route('/block', methods=['POST'])
def get_block():
    """
    根据block_info获取完整的区块对象的字节序列

    :return: 区块对象的字节序列
    """

    request_json = request.get_json()
    block_head_dict = request_json['block_head']
    block: Union[Block, None] = None

    # 从cursor_block中获取完整区块
    if status['blockchain'].cursor_block is not None \
            and status['blockchain'].cursor_block.block_head.block_height == block_head_dict["block_height"]:
        logger.debug("正在从cursor_block中获取完整区块")
        block = status['blockchain'].cursor_block

    # 从本地文件中获取完整区块
    elif status["blockchain"].get_block(block_head_dict["block_height"]) is not None:
        logger.debug("正在从本地文件中获取完整区块")
        block = status["blockchain"].get_block(block_head_dict["block_height"])

    # 从其他节点获取完整区块
    else:
        logger.debug("正在从其他节点获取完区块")
        rsp = requests.post('http://{node}/block'.format(node=['miner']), json=request_json)
        if rsp.status_code == 200:
            block = pickle.loads(codecs.decode(rsp.json()['block'].encode(), "base64"))
            with open("./blocks/federated_model" + block_head_dict["block_height"] + ".block", "wb") as f:
                f.write(pickle.dumps(block))

    # 验证区块哈希值是否满足
    valid: bool
    if hash_sha256(block.block_body) == block_head_dict['hash']:
        valid = True
    else:
        valid = False

    response = {
        # 先将 二进制 bytes 以 base64 的编码方式编码为 bytes ，再解码为 utf-8
        'block': codecs.encode(pickle.dumps(block), "base64").decode(),
        'valid': valid
    }
    return jsonify(response), 200


@app.route('/nodes/resolve', methods=["GET"])
def consensus():
    replaced = status['blockchain'].resolve_conflicts(STOP_EVENT)
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': status['blockchain'].hashchain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': status['blockchain'].hashchain
        }
    return jsonify(response), 200


@app.route('/stopmining', methods=['GET'])
def stop_mining():
    status['blockchain'].resolve_conflicts(STOP_EVENT)
    response = {
        'mex': "stopped!"
    }
    return jsonify(response), 200


def delete_prev_blocks() -> None:
    """
    删除之前的区块文件，根据文件后缀筛选

    :return:
    """
    files = glob.glob('blocks/*.block')
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='矿工监听的端口')
    parser.add_argument('-i', '--host', default='127.0.0.1', help='矿工的IP地址')
    parser.add_argument('-g', '--genesis', default=1, type=int, help='初始化创世区块')
    parser.add_argument('-l', '--update_limit', default=2, type=int, help='单个区块中最多包含多少个更新')
    parser.add_argument('-d', '--dataset_dir', default=".\\dataset", help='dataset数据存放文件夹')
    parser.add_argument('-b', '--block_dir', default=".\\block", help="区块文件存储位置")
    parser.add_argument('-m', '--miner_name', default="miner_1", help="矿工 name")
    parser.add_argument('-ma', '--maddress', help='其他矿工的IP端口')
    args = parser.parse_args()
    # 矿工地址
    address = "{host}:{port}".format(host=args.host, port=args.port)
    status['address'] = address
    if args.genesis == 0 and args.maddress is None:
        raise ValueError("Must set genesis=1 or specify maddress")
    delete_prev_blocks()

    # 如果该矿工为第一个矿工，则需初始化一个新的区块链
    if args.genesis == 1:
        logger.info("矿工:{} 为区块链中的首个矿工节点，开始初始化区块链设置".format(args.miner_name))
        model: Model = make_base(args.dataset_dir)
        logger.info("区块链中初始全局模型测试集准确率为:{}".format(model.accuracy))
        status['blockchain'] = Blockchain(miner_id=args.miner_name,
                                          block_dir=args.block_dir + path_separator + args.miner_name,
                                          update_limit=args.update_limit,
                                          dataset_dir=args.dataset_dir)
        logger.info("矿工:{} 开始 mining 创世区块".format(args.miner_name))
        mine(genesis_model=model)

    # 如果该矿工需要加入区块链，则需获取当前存在区块链，并向区块链中注册该矿工
    else:
        status['blockchain'] = Blockchain(miner_id=args.miner_name,
                                          block_dir=args.block_dir + path_separator + args.miner_name,
                                          dataset_dir=args.dataset_dir)
        status['blockchain'].register_node(args.maddress)
        requests.post('http://{node}/nodes/register'.format(node=args.maddress), json={'nodes': [address]})
        status['blockchain'].resolve_conflicts(STOP_EVENT)

    # 开启矿工服务
    logger.info("矿工:{} 开始在 {} 进行监听".format(args.miner_name, address))
    app.run(host=args.host, port=args.port)
