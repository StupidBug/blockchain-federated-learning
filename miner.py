"""
 - Blockchain for Federated Learning -
           Mining script
"""
import pickle

from blockchain import *
from threading import Thread, Event
from federatedlearner import *
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
    def __init__(self, stop_event, blockchain, node_identifier):
        self.stop_event = stop_event
        Thread.__init__(self)
        self.blockchain = blockchain
        self.node_identifier = node_identifier
        self.response = None

    def run(self):
        block, stopped = self.blockchain.proof_of_work(self.stop_event)
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


def mine():
    """
    挖矿

    :return:
    """

    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT, status["blockchain"], status["id"])
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
    # 区块链状态校验
    if status['s'] != "receiving":
        return 'Miner not receiving', 400

    # 参数合法性校验
    values = request.get_json()
    required = ['client', 'base_block_height', 'update', 'datasize', 'computing_time']
    if not all(k in values for k in required):
        return 'Missing values', 400
    if values['client'] in status['blockchain'].current_updates:
        return 'Model already stored', 400

    index = status['blockchain'].new_update(values['client'],
                                            values['base_block_height'],
                                            dict(pickle.loads(codecs.decode(values['model_updated'].encode(), "base64"))),
                                            values['datasize'],
                                            values['computing_time'])
    # 向所有miner节点转发该交易
    for node in status["blockchain"].node_addresses:
        requests.post('http://{node}/transactions/new'.format(node=node), json=request.get_json())

    # 交易合法性校验，成功则开始 mine
    if (status['s'] == 'receiving' and (
        len(status["blockchain"].current_updates) >= status['blockchain'].latest_block['update_limit'] or
            time.time()-status['blockchain'].latest_block['timestamp'] > status['blockchain'].latest_block['time_limit'])):
        mine()
    response = {'message': "Update will be added to block {index}".format(index=index)}
    return jsonify(response), 201


@app.route('/status', methods=['GET'])
def get_status():
    """
    获取 miner 状态

    :return 节点状态: [receiving | mining]
    """

    response = {
        'status': status['s'],
        'last_model_index': status['blockchain'].latest_block['block_height']
        }
    return jsonify(response), 200


@app.route('/chain', methods=['GET'])
def full_chain():
    """
    获取完整的区块链

    :return: 区块信息列表（区块链）
    """

    response = {
        'chain': status['blockchain'].hashchain,
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
    block_info = request_json['block_info']
    block: bytes = bytes()

    # 从cursor_block中获取完整区块
    if status['blockchain'].cursor_block.block_height == block_info["block_height"]:
        block = pickle.dumps(status['blockchain'].cursor_block)

    # 从本地文件中获取完整区块
    elif os.path.isfile("./blocks/federated_model" + block_info["block_height"] + ".block"):
        with open("./blocks/federated_model" + block_info["block_height"] + ".block", "rb") as f:
            block = f.read()

    # 从其他节点获取完整区块
    else:
        rsp = requests.post('http://{node}/block'.format(node=['miner']), json=request_json)
        if rsp.status_code == 200:
            block = rsp.json()['block']
            with open("./blocks/federated_model" + block_info["block_height"] + ".block", "wb") as f:
                f.write(block)

    # 验证区块哈希值是否满足
    valid: bool
    if hash_sha256(block) == block_info['hash']:
        valid = True
    else:
        valid = False

    response = {
        'block': block,
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
    parser.add_argument('-l', '--update_limit', default=10, type=int, help='单个区块中最多包含多少个更新')
    parser.add_argument('-d', '--dataset_dir', default="D:\\dataset", help='dataset数据存放文件夹')
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
        logger.info("矿工:{} 为区块链中的首个矿工节点，开始初始化区块链设置".format(address))
        model = make_base(args.dataset_dir)
        logger.info("区块链中初始全局模型测试集准确率为:{}".format(model.accuracy))
        status['blockchain'] = Blockchain(address, model, True, args.update_limit)

    # 如果该矿工需要加入区块链，则需获取当前存在区块链，并向区块链中注册该矿工
    else:
        status['blockchain'] = Blockchain(address)
        status['blockchain'].register_node(args.maddress)
        requests.post('http://{node}/nodes/register'.format(node=args.maddress), json={'nodes': [address]})
        status['blockchain'].resolve_conflicts(STOP_EVENT)

    # 开启矿工服务
    app.run(host=args.host, port=args.port)
