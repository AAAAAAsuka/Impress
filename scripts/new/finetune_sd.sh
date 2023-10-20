#bash
TRAIN_DIR=$1

export MODEL_NAME='stabilityai/stable-diffusion-2-1-base'


accelerate launch  train_text_to_image.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --train_data_dir=${TRAIN_DIR} \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=${batch_size} \
  --gradient_accumulation_steps=${grad_accum} \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=${step} \
  --learning_rate=5e-6 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --enable_xformers_memory_efficient_attention

