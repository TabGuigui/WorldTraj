export NAVSIM_EXP_ROOT="/data/worlddrive/navsim/exp/worldtraj"
export NUPLAN_MAPS_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/map"
export OPENSCENE_DATA_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/data/worlddrive/navsim"

TRAIN_TEST_SPLIT=navtrain
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1


python3  $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_worldtraj.py \
    agent=worldtraj_agent \
    agent.lr=3e-4 \
    agent.checkpoint_path="/data/worlddrive/navsim/exp/ckpt/cogvideox_tadwm_navsim_20wpt/tvar_10000.pkl" \
    agent.with_wm_proj=True \
    experiment_name=training_worlddrive \
    trainer.params.max_epochs=50  \
    dataloader.params.batch_size=32 \
    dataloader.params.num_workers=10 \
    train_test_split=navtrain \
    split=trainval \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=/mnt/gxt-share-navsim/dataset/trajworld_1024/worldtraj_train_1024_cache_v2 \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    agent.vocab_path="/data/diffusiondrive/trajectory_anchors_256.npy"\
    agent.cls_loss_weight=1 \
    agent.reg_loss_weight=20