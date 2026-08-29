
# GRPO训练
该阶段可以直接使用原数据，因为奖励函数仅通过答案打分，与思维链无关。

奖励函数设计在：src/rewards/zh_grammar_reward.py

模型训练指令：
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model your_model \
    --external_plugins src/rewards/zh_grammar_reward.py \
    --reward_funcs zh_gec_reward \
    --enable_thinking False \
    --use_vllm true \
    --template gemma \
    --lora_rank 8 \
    --lora_alpha 32 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 2000 \
    --sleep_level 1 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset processed_data/v1/nas_grpo.jsonl \
    --load_from_cache_file true \
    --max_length 1024  \
    --max_completion_length 768  \
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
    --output_dir output_grpo

# GLPO训练

## 数据预处理

```bash
python YuYi/glpo/prepare_glpo_data.py \
    --input  processed_data/grpo_train.jsonl \
    --output processed_data/grpo_train_M4.jsonl \
    --M 4
```

检查输出行数是原来的4倍。

## 启动训练

```bash
bash glpo/run_glpo.sh
```

# 推理 + 评测
通过--voting_samples 控制vote数量，设置为0时默认为greedy decoding。

python CSED_test.py \
    --model_path model \
    --input data/test.json \
    --output output_data.jsonl \
    --use_vllm \
    --max_retries 3 \
    --retry_temperature 1.0 \
    --cherrant_dir MuCGEC/scorers/ChERRANT \
    --save_errant_dir debug_errant/ \
    --voting_samples 0
    
python NaSGEC_test.py \
    --model_path model \
    --input data/test.json \
    --output output_data.jsonl \
    --reference_m2 data/NaSGEC-Exam/nasgec.exam.test.m2 \
    --use_vllm \
    --save_errant_dir debug_errant \
    --cherrant_dir MuCGEC/scorers/ChERRANT \
    --voting_temperature 1.0 \
    --voting_samples 0
