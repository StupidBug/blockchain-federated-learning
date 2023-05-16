client_num=3
dataset_dir="..\\dataset"
block_dir="..\\block"
updates_dir="..\\updates"
dataset_type="pathmnist"
learning_rate=0.0001
train_batch_size=16

python ../prepare.py -n=${client_num} -d=${dataset_dir} --dataset_type=${dataset_type}

# 启动 miner 节点
python ../miner.py --port=5000 --host="127.0.0.1" --genesis=1 --dataset_dir=${dataset_dir} --block_dir=${block_dir} --update_limit=${client_num} --dataset_type=${dataset_type}&

sleep 30

# 启动 client 节点
for i in $(seq 0 $((client_num - 1)));
do
  python ../client.py --miner=127.0.0.1:5000 --dataset_dir=${dataset_dir} --epochs=3 --common_rounds=500 --name="node_${i}" --updates_dir=${updates_dir} --learning_rate=${learning_rate} --dataset_type=${dataset_type} --train_batch_size=${train_batch_size}&
done