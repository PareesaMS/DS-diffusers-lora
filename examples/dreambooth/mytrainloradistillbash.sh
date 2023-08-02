export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/dog"
export OUTPUT_DIR="/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/model-lora-distill-2"

accelerate launch train_dreambooth_lora_distill.py \
	   --pretrained_model_name_or_path=$MODEL_NAME  \
	   --instance_data_dir=$INSTANCE_DIR \
	   --output_dir=$OUTPUT_DIR \
	   --instance_prompt="Dog over the moon" \
	   --resolution=512 \
	   --train_batch_size=1 \
	   --gradient_accumulation_steps=1 \
	   --checkpointing_steps=100 \
	   --learning_rate=1e-4 \
	   --lr_scheduler="constant" \
	   --lr_warmup_steps=0 \
	   --validation_prompt="A photo of sks dog in a bucket" \
	   --validation_epochs=50 \
	   --seed="0" \
	   --num_train_epochs=50
