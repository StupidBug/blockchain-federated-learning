#!/bin/bash

node_num=5
dataset_dir="D:\\dataset"

python ../prepare.py -n=$node_num -d=$dataset

for i in `seq 1, ${node_num}`;
do
  port = 5000 + ${i}
  python ../miner.py -p=$port -h=127.0.0.1 -g=${i} -ma=127.0.0.1:5000
done