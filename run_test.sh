#!/bin/bash

# 定义常量参数
BACKEND="openai"
HOST="127.0.0.1"
PORT="5000"
# 動態獲取 served-model-name
MODEL_NAME=$(ps aux | grep -E "python3 -m (vllm|vllm_ocisext).entrypoints.openai.api_server" | grep -oP "(?<=--served-model-name )[^\s]+" | head -n 1)
TOKENIZER="fake_model"
ENDPOINT="/v1/completions"
DATASET_NAME="random"
REQUEST_RATE="inf"
#RANDOM_INPUT_LEN="129024"
# 動態獲取 max-model-len 並設置為 RANDOM_INPUT_LEN，並減去 2048
MAX_MODEL_LEN=$(ps aux | grep -E "python3 -m (vllm|vllm_ocisext).entrypoints.openai.api_server" | grep -oP "(?<=--max-model-len )[^\s]+" | head -n 1)

# 檢查是否成功取得 MAX_MODEL_LEN 並進行減法操作
if [ -z "$MAX_MODEL_LEN" ]; then
  echo "無法取得 max-model-len，請確保該進程正在運行。"
  exit 1
fi

RANDOM_INPUT_LEN=$((MAX_MODEL_LEN - 2048))
RANDOM_OUTPUT_LEN="2048"

MEM_USAGE_OUT="output/mem_usage.txt"
ENGINE_PARAMS_OUT="output/engine_params.txt"

# 使用 nvidia-smi 提取 GPU 名稱
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)

# 將 GPU 名稱寫入到文件中
echo "GPU Name: $GPU_NAME" > $MEM_USAGE_OUT

# 將 warm up 註解和 Memory-Usage 追加到 mem_usage 檔案中
echo "Warm up status:" >> $MEM_USAGE_OUT
nvidia-smi | grep "MiB /" | awk '{print $9, $11}' >> $MEM_USAGE_OUT

# 顯示存檔成功的訊息
echo "Memory usage information (warm up status) has been appended to mem_usage."

# 將 engine params 註解和 vllm參數 追加到 engine_params 檔案中
echo "engine params:" > $ENGINE_PARAMS_OUT
ps aux | grep -E "python3 -m (vllm|vllm_ocisext).entrypoints.openai.api_server" | grep -oP "(--\S+ \S+|--\S+)" >> $ENGINE_PARAMS_OUT 

echo "參數已成功提取並保存到 params.txt 文件中。"

# 定义不同的 --num-prompts 参数
NUM_PROMPTS_LIST=(1 5 10)

# 循环执行三次，每次使用不同的 --num-prompts
for NUM_PROMPTS in "${NUM_PROMPTS_LIST[@]}"; do

    RESULT_FILE="${MODEL_NAME}_${RANDOM_INPUT_LEN}-${RANDOM_OUTPUT_LEN}_${NUM_PROMPTS}.json"
    echo "Running benchmark with --num-prompts=$NUM_PROMPTS"
    python3 benchmark_serving.py --backend "$BACKEND" \
                                 --host "$HOST" \
                                 --port "$PORT" \
                                 --model "$MODEL_NAME" \
				 --tokenizer "$TOKENIZER" \
                                 --endpoint "$ENDPOINT" \
                                 --dataset-name "$DATASET_NAME" \
                                 --request-rate "$REQUEST_RATE" \
                                 --num-prompts "$NUM_PROMPTS" \
                                 --random-input-len "$RANDOM_INPUT_LEN" \
                                 --random-output-len "$RANDOM_OUTPUT_LEN" \
    				 --save-result \
				 --result-dir "output" \
				 --result-filename "$RESULT_FILE"
done

NUM_PROMPTS_LIST=(32 64 128)
RANDOM_INPUT_LEN="200"
RANDOM_OUTPUT_LEN="250"
# 循环执行三次，每次使用不同的 --num-prompts
for NUM_PROMPTS in "${NUM_PROMPTS_LIST[@]}"; do
	
    RESULT_FILE="${MODEL_NAME}_${RANDOM_INPUT_LEN}-${RANDOM_OUTPUT_LEN}_${NUM_PROMPTS}.json"
    echo "Running benchmark with --num-prompts=$NUM_PROMPTS"
    python3 benchmark_serving.py --backend "$BACKEND" \
                                 --host "$HOST" \
                                 --port "$PORT" \
                                 --model "$MODEL_NAME" \
				 --tokenizer "$TOKENIZER" \
                                 --endpoint "$ENDPOINT" \
                                 --dataset-name "$DATASET_NAME" \
                                 --request-rate "$REQUEST_RATE" \
                                 --num-prompts "$NUM_PROMPTS" \
                                 --random-input-len "$RANDOM_INPUT_LEN" \
                                 --random-output-len "$RANDOM_OUTPUT_LEN" \
				 --save-result \
				 --result-dir "output" \
				 --result-filename "$RESULT_FILE"
done

NUM_PROMPTS_LIST=(1 4 8 16 32)
RANDOM_INPUT_LEN="100"
RANDOM_OUTPUT_LEN="100"
# 循环执行三次，每次使用不同的 --num-prompts
for NUM_PROMPTS in "${NUM_PROMPTS_LIST[@]}"; do
	
    RESULT_FILE="${MODEL_NAME}_${RANDOM_INPUT_LEN}-${RANDOM_OUTPUT_LEN}_${NUM_PROMPTS}.json"
    echo "Running benchmark with --num-prompts=$NUM_PROMPTS"
    python3 benchmark_serving.py --backend "$BACKEND" \
                                 --host "$HOST" \
                                 --port "$PORT" \
                                 --model "$MODEL_NAME" \
				 --tokenizer "$TOKENIZER" \
                                 --endpoint "$ENDPOINT" \
                                 --dataset-name "$DATASET_NAME" \
                                 --request-rate "$REQUEST_RATE" \
                                 --num-prompts "$NUM_PROMPTS" \
                                 --random-input-len "$RANDOM_INPUT_LEN" \
                                 --random-output-len "$RANDOM_OUTPUT_LEN" \
				 --save-result \
				 --result-dir "output" \
				 --result-filename "$RESULT_FILE"
done


# 將 After Benchmark 註解和 Memory-Usage 追加到 mem_usage 檔案中
echo "After Benchmark status:" >> $MEM_USAGE_OUT
nvidia-smi | grep "MiB /" | awk '{print $9, $11}' >> $MEM_USAGE_OUT

# 顯示存檔成功的訊息
echo "Memory usage information (After Benchmark status) has been appended to mem_usage."
