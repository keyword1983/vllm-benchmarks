# Benchmarking vLLM
## 1. 導覽
將目前目錄整包放到inference server內

## 2. 執行
執行bash `./run_test.sh` 用以測試text-only行為，執行bash `./run_vision_test.sh` 用以測試vision行為

## 3. 執行過程
執行過程中會有log顯示結果，可以看出各種統計的資訊：
### Serving Benchmark Result
* Successful requests:                     1
* Benchmark duration (s):                  87.72
* Total input tokens:                      14336
* Total generated tokens:                  2048
* Request throughput (req/s):              0.01
* Output token throughput (tok/s):         23.35
* Total Token throughput (tok/s):          186.77
#### Time to First Token
* Mean TTFT (ms):                          151.31
* Median TTFT (ms):                        151.31
* P99 TTFT (ms):                           151.31
#### Time per Output Token (excl. 1st token)
* Mean TPOT (ms):                          42.78
* Median TPOT (ms):                        42.78
* P99 TPOT (ms):                           42.78
#### Inter-token Latency
* Mean ITL (ms):                           42.78
* Median ITL (ms):                         42.76
* P99 ITL (ms):                            43.52

## 4. 查看output
output目錄中
* `engine_params.txt` 紀錄推論server的啟動參數
* `mem_usage.txt`     紀錄GPU的VRAM用量
* `${MODEL_NAME}_${RANDOM_INPUT_LEN}-${RANDOM_OUTPUT_LEN}_${NUM_PROMPTS}.json`  紀錄各種的input content長度與request數量的測試結果

## 5. 測試方案
目前會跑三種work load profile：
* 高：input長度為max-model-lens-2k output長度為2k，同時跑1/5/10個requests
* 中：input長度為200 output長度為250，同時跑32/64/128個requests
* 低：input長度為100 output長度為150，同時跑1/4/8/16/32個requests

## 6. 測試視覺行為
目前會跑兩種work load profile：
* a：input長度為50 output長度為300，同時跑8/16/32/64/128個requests（模擬簡短問句但是較長的影像內容總結）
* b：input長度為100 output長度為100，同時跑1/8/16/32個requests

## 7. 測試結果分析
測試結果會以json格式儲存於output目錄中，內容包括：
* `input_length`: 輸入內容長度
* `output_length`: 輸出內容長度
* `num_prompts`: 請求數量
* `successful_requests`: 成功請求數量
* `total_input_tokens`: 總輸入token數量
* `total_generated_tokens`: 總生成token數量
* `request_throughput`: 請求吞吐量（req/s）
* `output_token_throughput`: 輸出token吞吐量（tok/s）
* `total_token_throughput`: 總token吞吐量（tok/s）
* `time_to_first_token`: 首個token的等待時間（ms）
* `time_per_output_token`: 每個輸出token的等待時間（ms）
* `inter_token_latency`: token之間的延遲時間（ms）
