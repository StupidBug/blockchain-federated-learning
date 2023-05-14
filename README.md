# blockchain-federated-learning
Blockchain federated learning simulate by python (using pytorch)

## 结构设计

### 区块结构

区块的结构分为区块头和区块体:

#### 区块头结构

+ index : 区块在区块链中的高度
+ nonce : 挖出区块的nonce值
+ previous_hash : 上一个区块的哈希值
+ miner : 挖出该区块的矿工的身份标识
+ accuracy : 该区块链聚合得到的模型在测试集的准确度
+ f1_score : 该区块链聚合得到的模型在测试集的f1分数
+ timestamp: 区块生成时间
+ time_limit: 距离上个区块的时间间隔限制
+ update_limit: 单个区块中可提交的更新数量限制
+ hash: 区块体的哈希值

#### 区块体结构

+ model_updated : 梯度更新聚合后的模型
+ updates : client节点发送的梯度更新

## 数据集

默认提供Cifar10数据集和MedMnist数据集

## 如何开始?

1. 配置script文件夹中start.sh脚本中的参数
2. 执行脚本

