node_num=1
client_num=5
dataset_dir="D:\\dataset"

python ../prepare.py -n=${client_num} -d=${dataset_dir}

# 启动 miner 节点
for i in `seq 1 ${node_num}`;
do
  port=$((5000 + i))
  python ../miner.py -p=${port} -h=127.0.0.1 -g=${i} -ma=127.0.0.1:5000
done

# 启动 client 节点
for i in `seq 1 ${client_num}`;
do
  python ../client.py -m=127.0.0.1 -d=${dataset_dir} -e=10 -n=node_${i}
done