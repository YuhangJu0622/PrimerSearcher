#!/bin/bash

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model /root/autodl-tmp/modelscope_cache/Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --deepspeed zero3 \
    --tuner_type lora \
    --dataset /root/Project/openseeker_train_122880_clean.jsonl \
    --val_dataset /root/Project/openseeker_eval_122880_clean.jsonl \
    --eval_strategy steps \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj \
    --router_aux_loss_coef 0 \
    --max_length 122880 \
    --loss_scale ignore_empty_think \
    --lazy_tokenize false \
    --attn_impl flash_attn \
    --packing true \
    --padding_free true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name qwen3-30b-a3b-lora-bf16-2gpu-rtx6000-122k-exp6 \
    --output_dir /root/autodl-tmp/output/qwen3_30b_lora_2gpu_rtx6000_122k_exp6 \
    --eval_steps 50 \
    --save_steps 20 \
    --save_total_limit 50 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_liger_kernel true
