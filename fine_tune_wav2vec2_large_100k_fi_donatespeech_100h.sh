#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p gpu-nvlink,dgx-spa
#SBATCH --time=4-23:55:00
#SBATCH -J gpu_job_1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/work/getmany1/wav2vec/fine_tune_wav2vec2_large_100k_fi_donatespeech_100h_SEGMENTED.out

module purge
module load sox/14.4.2
module load anaconda/2021-03-tf2
module load cuda/11.2.1

conda init
source activate /scratch/work/getmany1/wav2vec/w2venv

python /scratch/work/getmany1/wav2vec/run_asr_fi_donatespeech_100h.py \
--output_dir="/scratch/work/getmany1/wav2vec/wav2vec2-fi-donatespeech_100h" \
--cache_dir="/scratch/elec/puhe/p/getmany1/cache" \
--num_train_epochs="80" \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--gradient_accumulation_steps="48" \
--evaluation_strategy="steps" \
--save_total_limit="20" \
--save_steps="740" \
--eval_steps="740" \
--logging_steps="200" \
--learning_rate="5e-4" \
--warmup_steps="7400" \
--model_name_or_path="/scratch/work/getmany1/wav2vec/wav2vec2_large_100k_with_SWE_vocab_13_03_2021" \
--target_feature_extractor_sampling_rate \
--dataset_name="donatespeech" \
--train_split_name="train" \
--validation_split_name="train" \
--orthography timit \
--preprocessing_num_workers="$(nproc)" \
--group_by_length \
--freeze_feature_extractor \
--verbose_logging \
--gradient_checkpointing \
--fp16
