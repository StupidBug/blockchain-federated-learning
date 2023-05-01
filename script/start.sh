client_num=2
dataset_dir="..\\dataset"
block_dir="..\\block"
updates_dir="..\\updates"

python ../prepare.py -n=${client_num} -d=${dataset_dir}

# 启动 miner 节点
python ../miner.py --port=5000 --host="127.0.0.1" --genesis=1 --dataset_dir=${dataset_dir} --block_dir=${block_dir} --update_limit=1 &

sleep 10

# 启动 client 节点
for i in $(seq 0 $((client_num - 1)));
do
  python ../client.py --miner=127.0.0.1:5000 --dataset_dir=${dataset_dir} --epochs=5 --common_rounds=50 --name="node_${i}" --updates_dir=${updates_dir} &
done