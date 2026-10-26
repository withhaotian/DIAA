MODEL_PATH=shuyuej/Llama-3.2-3B-Instruct-GPTQ # meta-llama/Llama-3.2-1B-Instruct #shuyuej/Llama-3.2-3B-Instruct-GPTQ shuyuej/vicuna-7b-v1.5-GPTQ
MODEL_NAME=Llama-3.2-3B-Instruct-GPTQ # Llama-3.2-1B-Instruct # Llama-3.2-3B-Instruct-GPTQ
CACHE_LUT_PATH=("Llama-3.2-3B-Instruct_topk") # ("Llama-3.2-3B-Instruct_topk") ("vicuna-7b-v1.5_topk")

TEMP=0.0  # greedy decoding
DATA_NUM=20
SEED=2026
MAX_NEW_TOKENS=512
MAX_OUTPUT_LENGTH=3000  # maximum preallocated KV cache

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

GPU_DEVICES=0
IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

TASK_NAME=("sum" "math" "mt_bench" "qa" "rag" "translation")
BENCH_NAME="spec_bench"

torch_dtype="float16"

HF_MIRROR=https://hf-mirror.com

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Current model: $MODEL_PATH"

for task in ${TASK_NAME[@]}; do
echo "Evaluating task: $task"...

  echo "============================="
  echo "||        Vanilla          ||"
  echo "============================="
  HF_ENDPOINT=${HF_MIRROR} CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path ${MODEL_PATH} --model-id ${MODEL_NAME} --bench-name ${BENCH_NAME} --max-new-tokens ${MAX_NEW_TOKENS}  --max-output-length ${MAX_OUTPUT_LENGTH} --task-name $task --data-num ${DATA_NUM} --temperature ${TEMP} --seed ${SEED} --dtype $torch_dtype --num-gpus-per-model ${NUM_GPUS} --num-gpus-total ${NUM_GPUS} --timestamp ${TIMESTAMP}

  echo "============================="
  echo "||          DIAA           ||"
  echo "============================="
  HF_ENDPOINT=${HF_MIRROR} CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_diaa --model-path ${MODEL_PATH} --model-id ${MODEL_NAME}  --bench-name ${BENCH_NAME} \
  --temperature ${TEMP} --dtype $torch_dtype --task-name $task --data-num ${DATA_NUM} --max-new-tokens ${MAX_NEW_TOKENS} --max-output-length ${MAX_OUTPUT_LENGTH} --num-gpus-per-model ${NUM_GPUS} --num-gpus-total ${NUM_GPUS} --seed ${SEED} --cache-lut --cache-lut-path ${CACHE_LUT_PATH} --tree_version dynamic --tree_budget 50 --max_depth 7 --min_thres 1e-5 --dy_interval 100 --timestamp ${TIMESTAMP}

done
