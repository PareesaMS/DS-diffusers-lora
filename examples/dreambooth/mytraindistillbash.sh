#export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/dog"
export OUTPUT_DIR="/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/model-lora"

accelerate launch train_dreambooth_distill.py \
	   --pretrained_model_name_or_path=$MODEL_NAME  \
	   --instance_data_dir=$INSTANCE_DIR \
	   --output_dir=$OUTPUT_DIR \
	   --instance_prompt="Dog over the moon" \
	   --resolution=512 \
	   --train_batch_size=1 \
	   --gradient_accumulation_steps=1 \
	   --learning_rate=5e-6 \
	   --lr_scheduler="constant" \
	   --lr_warmup_steps=0
