#conda init bash
#source /usr/local/lib/miniconda3/bin/activate spider
#conda env list
export PYTHONPATH="${PWD}:$PYTHONPATH"
# 修复：当前目录就是项目根目录，不用CODE_ROOT变量
cp ./petreloss.conf /root/petreloss.conf

# 昇腾NPU机器，删除nvidia-smi探测，手动填写NPU卡数量
# GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# echo 'export GPU_COUNT=${GPU_COUNT}'
GPU_COUNT=1  # 多卡改成 2/4/8，根据你的910C卡数修改
# enable rdma 多机通信参数，昇腾NPU分布式可用
# 如下 3 个变量的默认值不满足需求时可在训练脚本启动之前覆盖对应的值
NCCL_DEBUG=${NCCL_DEBUG:-INFO}
NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23}
NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
# 如下变量建议用户直接使用默认值，不要自行修改
# NCCL_IB_HCA=<平台根据实例规格自动注入>
# NCCL_IB_DISABLE=0
# NCCL_IB_GID_INDEX=<平台根据实例规格自动注入>
# NCCL_SOCKET_IFNAME=<平台注入默认的VPC网卡>
# NCCL_IB_PCI_RELAXED_ORDERING=1
# NCCL_TOPO_FILE=<平台根据实例规格自动注入>
#######################################
# 多NPU分布式训练



 #!/usr/bin/env bash

TRAIN_AS_DEPLOY=True MODE=multi GPUS=${GPU_COUNT} MASTER_PORT=${MASTER_PORT:-29507} bash tools/ddp_train.sh tools/train_spetr.py $1




# 单NPU训练，使用时取消注释上面一行，注释下面multi
# TRAIN_AS_DEPLOY=True MODE=single GPUS=${GPU_COUNT} MASTER_PORT=${MASTER_PORT:-29507} bash tools/ddp_train.sh tools/train_spetr.py $1
