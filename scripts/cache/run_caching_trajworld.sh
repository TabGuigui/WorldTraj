TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAPS_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/map"
export OPENSCENE_DATA_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1"
export NAVSIM_EXP_ROOT="/mnt/gxt-share-navsim/dataset/worldtraj_release"
export NAVSIM_DEVKIT_ROOT="/data/worlddrive/navsim"

CACHE_PATH=$NAVSIM_EXP_ROOT/worldtraj_train_cache

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
echo "NAVSIM_DEVKIT_ROOT ${NAVSIM_DEVKIT_ROOT}"

torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=worldtraj_agent \
    experiment_name=trajworld_agent_cache \
    agent.cache_vae_token=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    agent.vocab_path="anchors/trajectory_anchors_256.npy" \
    agent.sim_reward_path="anchors/formatted_pdm_score_256.npy" \
    agent.pretrained_vae_path="/data/diffusiondrive/ckpt/zai-org/CogVideoX-2b"

