client_num=2
dataset_dir="..\\dataset"
block_dir="..\\block"
updates_dir="..\\updates"

python ../prepare.py -n=${client_num} -d=${dataset_dir}

# 启动 miner 节点
python ../miner.py -p=5000 -i="127.0.0.1" -g=1 -d=${dataset_dir} -b=${block_dir} &

sleep 10

# 启动 client 节点
for i in $(seq 0 $((client_num - 1)));
do
  python ../client.py -m=127.0.0.1:5000 -d=${dataset_dir} -e=10 -n="node_${i}" -u=${updates_dir} &
done