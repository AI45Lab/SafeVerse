set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_START_METHOD=spawn

export SERVER_IP="the server ip"

export SERVER_PORT="8000"
echo "Please export the server ip!"

# 调整NCCL报告等级，INFO是最完整的
export NCCL_DEBUG="WARN"

export MAX_RESETTING_ENV_COUNT=16

ray start --head --dashboard-host=0.0.0.0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

BASE_NAME="your project name"
PROJECT_NAME="${BASE_NAME}"
EXPERIMENT_NAME="${BASE_NAME}"
WANDB_SAVE_DIR="${BASE_NAME}"

export WORLD_SIZE=1


export SAVE_CHECKPOINT_DIR="checkpoint saving dir"

REF_MODEL_PATH="base model path"
DATA="data/data.parquet"
MINI_BACTH_SIZE=32
MICRO_BS_PER_GPU=4

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.image_key=null \
    data.prompt_key=null \
    data.train_files=[${DATA}] \
    data.val_files=[${DATA}] \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=10240 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BACTH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BS_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BS_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key="MCSimulator" \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=512 \
    actor_rollout_ref.rollout.agent.max_turns=64 \
    actor_rollout_ref.rollout.agent.segment_len=1 \
    actor_rollout_ref.rollout.agent.concurrent_workers=8 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=50 \
    trainer.test_freq=10000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${E_XPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=1000 2>&1 | tee ./logs/${WANDB_SAVE_DIR}.log \