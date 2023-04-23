"""
 - Blockchain for Federated Learning -
           Mining script
"""

from blockchain import *
from threading import Thread, Event
from federatedlearner import *
from datasets import GlobalDataset, NodeDataset
from model import *
import codecs
import os
import glob
import log

"""设置全局日志"""
logger = log.setup_custom_logger("miner")


def make_base():
    """
    Function to do the base level training on the first set of client data
    for the genesis block
    """
    # TODO 是否使用多种模型（看进度）
    net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)

    global_dataset = GlobalDataset(root="/tmp/dataset", train=False, )
    worker = NNWorker(dataset["train_images"],
        dataset["train_labels"],
        dataset["test_images"],
        dataset["test_labels"],
        0,
        "base0")
    worker.build_base()
    model = dict()
    model['model'] = worker.get_model()
    model['accuracy'] = worker.evaluate()
    worker.close()
    return model


class PoWThread(Thread):
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

# 区块链节点状态
status = {
    's': "receiving",
    'id': str(uuid4()).replace('-', ''),        # 节点ID
    'blockchain': None,                         #
    'address': ""                               # 节点IP和Port
    }


def mine():
    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT,status["blockchain"], status["id"])
    status['s'] = "mining"
    thread.start()


def on_end_mining(stopped):
    if status['s'] == "receiving":
        return
    if stopped:
        status["blockchain"].resolve_conflicts(STOP_EVENT)
    status['s'] = "receiving"
    for node in status["blockchain"].nodes:
        requests.get('http://{node}/stopmining'.format(node=node))


@app.route('/transactions/new',methods=['POST'])
def new_transaction():
    """
    处理新的一笔交易
    :return:
    """
    if status['s'] != "receiving":
        return 'Miner not receiving', 400
    values = request.get_json()

    required = ['client','baseindex','update','datasize','computing_time']
    if not all(k in values for k in required):
        return 'Missing values', 400
    if values['client'] in status['blockchain'].current_updates:
        return 'Model already stored', 400
    index = status['blockchain'].new_update(values['client'],
        values['baseindex'],
        dict(pickle.loads(codecs.decode(values['update'].encode(), "base64"))),
        values['datasize'],
        values['computing_time'])
    for node in status["blockchain"].nodes:
        requests.post('http://{node}/transactions/new'.format(node=node), json=request.get_json())
    if (status['s'] == 'receiving' and (
        len(status["blockchain"].current_updates)>=status['blockchain'].last_block['update_limit']
        or time.time()-status['blockchain'].last_block['timestamp']>status['blockchain'].last_block['time_limit'])):
        mine()
    response = {'message': "Update will be added to block {index}".format(index=index)}
    return jsonify(response), 201


@app.route('/status',methods=['GET'])
def get_status():
    """
    获取 miner 状态
    :return:
    """
    response = {
        'status': status['s'],
        'last_model_index': status['blockchain'].last_block['index']
        }
    return jsonify(response), 200


@app.route('/chain', methods=['GET'])
def full_chain():
    """
    获取完整的区块链
    :return:
    """
    response = {
        'chain': status['blockchain'].hashchain,
        'length': len(status['blockchain'].hashchain)
    }
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    """
    注册节点
    :return:
    """
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Enter valid nodes in the list ", 400
    for node in nodes:
        if node != status['address'] and node not in status['blockchain'].nodes:
            status['blockchain'].register_node(node)
            for miner in status['blockchain'].nodes:
                if miner != node:
                    print("node",node,"miner",miner)
                    requests.post('http://{miner}/nodes/register'.format(miner=miner),
                        json={'nodes': [node]})
    response = {
        'message': "New nodes have been added",
        'total_nodes': list(status['blockchain'].nodes)
    }
    return jsonify(response), 201


@app.route('/block', methods=['POST'])
def get_block():
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block", "rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']), json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block", "wb") as f:
                    pickle.dump(block, f)
    valid = False
    if Blockchain.hash(str(block)) == hblock['hash']:
        valid = True
    response = {
        'block': str(block),
        'valid': valid
    }
    return jsonify(response), 200


@app.route('/model', methods=['POST'])
def get_model():
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block", "rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']), json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block", "wb") as f:
                    pickle.dump(block, f)
    valid = False
    model = block.basemodel
    if Blockchain.hash(codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode()) == hblock['model_hash']:
        valid = True
    response = {
        'model': codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode(),
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
    return jsonify(response) ,200


def delete_prev_blocks():
    files = glob.glob('blocks/*.block')
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    parser.add_argument('-i', '--host', default='127.0.0.1', help='IP address of this miner')
    parser.add_argument('-g', '--genesis', default=0, type=int, help='instantiate genesis block')
    parser.add_argument('-l', '--ulimit', default=10, type=int, help='number of updates stored in one block')
    parser.add_argument('-ma', '--maddress', help='other miner IP:port')
    args = parser.parse_args()
    # 矿工地址
    address = "{host}:{port}".format(host=args.host, port=args.port)
    status['address'] = address
    if args.genesis == 0 and args.maddress is None:
        raise ValueError("Must set genesis=1 or specify maddress")
    delete_prev_blocks()
    # 初始化创世区块
    if args.genesis == 1:
        model = make_base()
        logger.info("base model accuracy:", model['accuracy'])
        status['blockchain'] = Blockchain(address, model, True, args.ulimit)
    else:
        status['blockchain'] = Blockchain(address)
        status['blockchain'].register_node(args.maddress)
        requests.post('http://{node}/nodes/register'.format(node=args.maddress), json={'nodes': [address]})
        status['blockchain'].resolve_conflicts(STOP_EVENT)
    app.run(host=args.host,port=args.port)
