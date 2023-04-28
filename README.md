# blockchain-federated-learning
Blockchain federated learning simulate by python


### 结构设计

#### 区块结构

区块的结构以字典形式进行保存，具体键值如下:

+ index : 区块在区块链中的高度
+ nonce : 挖出区块的nonce值
+ previous_hash : 上一个区块的哈希值
+ miner : 挖出该区块的矿工的身份标识
+ accuracy : 该区块链聚合得到的模型在测试集的准确度
+ timestamp: 区块生成时间
+ time_limit: 距离上个区块的时间间隔限制
+ update_limit: 单个区块中可提交的更新数量限制
+ model_hash: 模型的哈希值

#### 区块链

区块链的形式为区块字典构成的列表

### 如何开始?

1. 配置script文件夹中start.sh脚本中的参数
2. 执行脚本

Unfinished