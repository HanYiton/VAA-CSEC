#!/bin/bash

# ===== GLPO 超参 =====
export GLPO_ALPHA=${GLPO_ALPHA:-0.5}
export GLPO_M=${GLPO_M:-6}
export GLPO_K=${GLPO_K:-8}
export GLPO_STD_SCALE=${GLPO_STD_SCALE:-1}
export GLPO_DEBUG=${GLPO_DEBUG:-1}
export GLPO_GROUP_COL_IDX=${GLPO_GROUP_COL_IDX:-1}

EXP_NAME="glpo_M${GLPO_M}_alpha${GLPO_ALPHA}"
OUTPUT_DIR="output_${EXP_NAME}"
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model model \
    --external_plugins glpo_swift_plugin.py src/rewards/zh_grammar_reward.py \
    --reward_funcs zh_gec_reward zh_gec_group_reward \
    --reward_weights 1.0 1.0 \
    --enable_thinking True \
    --use_vllm true \
    --template qwen3 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 1000 \
    --sleep_level 1 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset grpo_train.jsonl \
    --load_from_cache_file false \
    --dataset_shuffle true \
    --max_length 256 \
    --max_completion_length 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --save_steps 200 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --log_completions true \
    --report_to wandb \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --log_entropy true \
    --beta 0.001 \
    --advantage_estimator grpo \
    --output_dir ${OUTPUT_DIR}
