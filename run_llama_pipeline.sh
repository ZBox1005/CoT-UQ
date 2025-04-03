export PYTHONPATH=./
MODEL_ENGINE=$1
UQ_ENGINE=$2
DATASET=$3
OUTPUT_PATH=$4

TEMP=1.0
TRY_TIMES=20

CUDA_VISIBLE_DEVICES='0' \
python inference_refining.py --dataset ${DATASET} --model_engine ${MODEL_ENGINE} --model_path ${MODEL_ENGINE} --temperature ${TEMP} --output_path ${OUTPUT_PATH} --try_times ${TRY_TIMES}

CUDA_VISIBLE_DEVICES='0' \
python stepuq.py --dataset ${DATASET} --uq_engine ${UQ_ENGINE} --model_path ${MODEL_ENGINE} --temperature ${TEMP} --output_path ${OUTPUT_PATH} --try_times 5