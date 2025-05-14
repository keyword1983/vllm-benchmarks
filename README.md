# Benchmarking vLLM


1. 將目前目錄整包放到inference server 內
2. 將要測試的model的 tokenizer.json 等json放入 fake_model 目錄下
3. 執bash ./run_test.sh 用以測試text-only行為,執行bash ./run_vision_test.sh 用以測試vision行為
執行過程中會有 log 顯示結果, 可以看出各種統計的資訊:
============ Serving Benchmark Result ============
Successful requests:                     1
Benchmark duration (s):                  87.72
Total input tokens:                      14336
Total generated tokens:                  2048
Request throughput (req/s):              0.01
Output token throughput (tok/s):         23.35
Total Token throughput (tok/s):          186.77
---------------Time to First Token----------------
Mean TTFT (ms):                          151.31
Median TTFT (ms):                        151.31
P99 TTFT (ms):                           151.31
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.78
Median TPOT (ms):                        42.78
P99 TPOT (ms):                           42.78
---------------Inter-token Latency----------------
Mean ITL (ms):                           42.78
Median ITL (ms):                         42.76
P99 ITL (ms):                            43.52
==================================================
測試會跑多個round 配合各種的input content長度與request數量的測試組合.
目前run_test.sh 會根據目前model的啟動參數的max-model-lens 來決定最大的 input content長度.
一共會跑三種work load profile:高中低
高: input 長度為max-model-lens-2k output長度為2k  ,同時跑 1/5/10 個requests
中: input 長度為200 output長度為250  ,同時跑 32/64/128 個requests
低: input 長度為100 output長度為150  ,同時跑 1/4/8/16/32 個requests
目前run_vision_test.sh 會根據目前model的啟動參數的max-model-lens 來決定最大的 input content長度.
且動態產生指定尺寸的雜訊image來當輸入token測試, image的尺寸在測試的時候作為參數輸入bash ./run_vision_test.sh IM_WIDTH IM_HEIGHT , 預設為512, 512
一共會跑兩種work load profile:
1: input 長度為50 output長度為300  ,同時跑 8/16/32/64/128 個requests (模擬簡短問句但是較長的影像內容總結)
2: input 長度為100 output長度為100  ,同時跑 1/8/16/32 個requests



4. 查看output 中的benchmark結果.

output目錄中
engine_params.txt 紀錄推論server的啟動參數
mem_usage.txt     紀錄GPU 的VRAM用量
${MODEL_NAME}_${RANDOM_INPUT_LEN}-${RANDOM_OUTPUT_LEN}_${NUM_PROMPTS}.json  紀錄各種的input content長度與request數量的測試結果


