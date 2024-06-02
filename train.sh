export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=6003
export NCCL_P2P_DISABLE=1
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node=8 \
    --master_port=29504 \
    experiments/train_se3_diffusion.py \
    --config-name=train
