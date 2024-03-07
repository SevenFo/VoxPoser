export VALIDATION_WORKERS=4
export MAX_CONCURRENT_REQUESTS=1
export MAX_BEST_OF=1 # This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time, and return the best in terms of overall log probability over the entire generated sequence [env: MAX_BEST_OF=] [default: 2]
export MAX_INPUT_LENGTH=2048
export MAX_BATCH_PREFILL_TOKENS=2048
export MAX_TOTAL_TOKENS=2560
export QUANTIZE=awq
export PORT=40906

text-generation-launcher --model-id deepseek-coder-6.7B-instruct-AWQ