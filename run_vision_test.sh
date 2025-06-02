#!/bin/bash

# 定义常量参数
BACKEND="openai-chat"
HOST="127.0.0.1"
#HOST="10.103.10.74"
PORT="5000"

#!/bin/bash

# 檢查 jq 是否存在
if ! command -v jq &> /dev/null; then
    echo "❗ jq 未安裝，正在進行安裝..."

    # 檢查 OS 類型並選擇適當的套件管理器
    if [ -f /etc/debian_version ]; then
        echo "🔍 偵測到 Debian/Ubuntu 系統，使用 apt 安裝 jq..."
        apt update && apt install -y jq
    elif [ -f /etc/redhat-release ]; then
        echo "🔍 偵測到 RHEL/CentOS 系統，使用 yum 安裝 jq..."
        yum install -y epel-release && yum install -y jq
    elif command -v apk &> /dev/null; then
        echo "🔍 偵測到 Alpine Linux，使用 apk 安裝 jq..."
        apk add --no-cache jq
    else
        echo "🚫 無法自動判斷套件管理器，請手動安裝 jq。"
        exit 1
    fi

    # 再次檢查 jq 是否成功安裝
    if command -v jq &> /dev/null; then
        echo "✅ jq 安裝完成。"
    else
        echo "❌ jq 安裝失敗，請手動處理。"
        exit 1
    fi
else
    echo "✅ jq 已安裝：$(jq --version)"
fi


# 動態獲取 served-model-name
response=$(curl -s http://${HOST}:${PORT}/v1/models)
MODEL_NAME=$(echo "$response" | jq -r '.data[0].id')
#MODEL_NAME=$(ps aux | grep -E "python3 -m (vllm|vllm_ocisext).entrypoints.openai.api_server" | grep -oP "(?<=--served-model-name )[^\s]+" | head -n 1)
#TOKENIZER="fake_model"
TOKENIZER=$(echo "$response" | jq -r '.data[0].root')
ENDPOINT="/v1/chat/completions"
DATASET_NAME="ocisvision"
REQUEST_RATE="inf"
IM_WIDTH="512"
IM_HEIGHT="512"
# 使用传入的参数设置 IM_WIDTH 和 IM_HEIGHT，默认值为 512
IM_WIDTH="${1:-512}"
IM_HEIGHT="${2:-512}"

echo "Image Width: $IM_WIDTH"
echo "Image Height: $IM_HEIGHT"

#RANDOM_INPUT_LEN="129024"
# 動態獲取 max-model-len 並設置為 RANDOM_INPUT_LEN，並減去 2048
MAX_MODEL_LEN=$(ps aux | grep -E "python3 -m (vllm|vllm_ocisext).entrypoints.openai.api_server" | sed -n 's/.*--max-model-len[= ]\([0-9]\+\).*/\1/p' | head -n 1)

# 檢查是否成功取得 MAX_MODEL_LEN 並進行減法操作
if [ -z "$MAX_MODEL_LEN" ]; then
  echo "無法取得 max-model-len，請確保該進程正在運行。"
#  exit 1
fi

#RANDOM_INPUT_LEN=$((MAX_MODEL_LEN - 2048))
#RANDOM_OUTPUT_LEN="2048"

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
NUM_PROMPTS_LIST=(8 16 32 64 128)
OV_INPUT_LEN="50"
OV_OUTPUT_LEN="300"

# 循环执行三次，每次使用不同的 --num-prompts
for NUM_PROMPTS in "${NUM_PROMPTS_LIST[@]}"; do

    RESULT_FILE="${MODEL_NAME}_${OV_INPUT_LEN}-${OV_OUTPUT_LEN}_${NUM_PROMPTS}.json"
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
                                 --ov-input-len "$OV_INPUT_LEN" \
                                 --ov-output-len "$OV_OUTPUT_LEN" \
				 --im-width "$IM_WIDTH" \
				 --im-height "$IM_HEIGHT" \
    				 --save-result \
				 --result-dir "output" \
				 --result-filename "$RESULT_FILE"
done

NUM_PROMPTS_LIST=(1 8 16 32)
OV_INPUT_LEN="100"
OV_OUTPUT_LEN="100"

# 循环执行三次，每次使用不同的 --num-prompts
for NUM_PROMPTS in "${NUM_PROMPTS_LIST[@]}"; do

    RESULT_FILE="${MODEL_NAME}_${OV_INPUT_LEN}-${OV_OUTPUT_LEN}_${NUM_PROMPTS}.json"
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
                                 --ov-input-len "$OV_INPUT_LEN" \
                                 --ov-output-len "$OV_OUTPUT_LEN" \
				 --im-width "$IM_WIDTH" \
				 --im-height "$IM_HEIGHT" \
    				 --save-result \
				 --result-dir "output" \
				 --result-filename "$RESULT_FILE"
done


# 將 After Benchmark 註解和 Memory-Usage 追加到 mem_usage 檔案中
echo "After Benchmark status:" >> $MEM_USAGE_OUT
nvidia-smi | grep "MiB /" | awk '{print $9, $11}' >> $MEM_USAGE_OUT

# 顯示存檔成功的訊息
echo "Memory usage information (After Benchmark status) has been appended to mem_usage."
