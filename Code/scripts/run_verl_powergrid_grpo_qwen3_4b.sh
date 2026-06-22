#!/usr/bin/env bash
set -euo pipefail

# GRPO/RLVR training for the IEEE14 M1+M2+EMT tool-agent task.
# Run from Code/ or set CODE_DIR explicitly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VERL_DIR="${VERL_DIR:-$(cd "${CODE_DIR}/../verl-main" 2>/dev/null && pwd || true)}"

if [[ -z "${VERL_DIR}" || ! -d "${VERL_DIR}/verl" ]]; then
  echo "Could not find verl-main. Set VERL_DIR=/path/to/verl-main." >&2
  exit 1
fi

export PYTHONPATH="${CODE_DIR}:${VERL_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"

MODEL_PATH="${MODEL_PATH:-/nas/models/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-${CODE_DIR}/verl_data/powergrid_ieee14_emt_seed20260610/train.parquet}"
VAL_FILE="${VAL_FILE:-${CODE_DIR}/verl_data/powergrid_ieee14_emt_seed20260610/val.parquet}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${CODE_DIR}/config/verl_powergrid_tool_config.yaml}"
REWARD_PATH="${REWARD_PATH:-${CODE_DIR}/gridmind_mini/verl_reward.py}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TOOL_PARSER_FORMAT="${TOOL_PARSER_FORMAT:-hermes}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"

PROJECT_NAME="${PROJECT_NAME:-powergym_verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_ieee14_m1m2emt_grpo}"
LOGGER="${LOGGER:-[\"console\",\"wandb\"]}"

if [[ "${SAVE_FREQ}" == "final" || "${SAVE_FREQ}" == "FINAL" || "${SAVE_FREQ}" == "last" || "${SAVE_FREQ}" == "LAST" ]]; then
  # verl only saves the last step when save_freq is positive. Use a large interval
  # to suppress periodic checkpoints while preserving the final checkpoint.
  SAVE_FREQ_VALUE=1000000000
  SAVE_FREQ_DISPLAY="final-only"
else
  SAVE_FREQ_VALUE="${SAVE_FREQ}"
  SAVE_FREQ_DISPLAY="${SAVE_FREQ}"
fi

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  echo "Missing train/val parquet files." >&2
  echo "Generate them first:" >&2
  echo "  PYTHONPATH=${CODE_DIR} python3 ${CODE_DIR}/scripts/export_verl_powergrid_dataset.py" >&2
  exit 1
fi

if [[ ! -f "${TOOL_CONFIG_PATH}" ]]; then
  echo "Missing tool config: ${TOOL_CONFIG_PATH}" >&2
  exit 1
fi

echo "Starting verl GRPO PowerGrid training"
echo "Code dir: ${CODE_DIR}"
echo "verl dir: ${VERL_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Train: ${TRAIN_FILE}"
echo "Val: ${VAL_FILE}"
echo "Tool config: ${TOOL_CONFIG_PATH}"
echo "Reward: ${REWARD_PATH}"
echo "HF attention implementation: ${ATTN_IMPLEMENTATION}"
echo "vLLM attention backend: ${VLLM_ATTENTION_BACKEND}"
echo "Rollout max model len: ${ROLLOUT_MAX_MODEL_LEN}"
echo "Rollout enforce eager: ${ROLLOUT_ENFORCE_EAGER}"
echo "Tool parser format: ${TOOL_PARSER_FORMAT}"
echo "Save frequency: ${SAVE_FREQ_DISPLAY}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.return_multi_modal_inputs=False \
  data.apply_chat_template_kwargs='{}' \
  reward.custom_reward_function.path="${REWARD_PATH}" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9000 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=9000 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=9000 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER_FORMAT}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=6000 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ_VALUE}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"
