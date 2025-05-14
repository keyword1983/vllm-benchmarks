# Benchmarking vLLM


1. 將目前目錄整包放到inference server 內
2. 將要測試的model的 tokenizer.json 等json放入 fake_model 目錄下
3. 執行run_test.sh 用以測試text-only行為,執行run_vision_test.sh 用以測試vision行為
4. 查看output 中的benchmark結果. 
