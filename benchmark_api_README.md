# vLLM Benchmark API Server

This API server provides HTTP endpoints to run vLLM benchmarks, allowing you to trigger benchmarks remotely and retrieve results through a REST API.

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- vLLM and its dependencies

## Installation

Make sure you have all the required packages installed:

```bash
pip install fastapi uvicorn pydantic
```

vLLM should already be installed in your environment.

## Running the Server

### Direct Method

Start the API server with:

```bash
# Make the script executable
chmod +x start_benchmark_api.sh

# Run the server
./start_benchmark_api.sh --host 0.0.0.0 --port 8080
```

### Docker Method

You can also run the server in a Docker container:

```bash
# Make the script executable
chmod +x run_benchmark_api_docker.sh

# Run the server with Docker
./run_benchmark_api_docker.sh --models-path /path/to/your/models --port 8080
```

This will build a Docker image and start a container with the API server. The models directory will be mounted into the container.

## Accessing the Server

The server will be available at http://localhost:8080 (or the host/port you specified).

API documentation is automatically generated and available at http://localhost:8080/docs

You can use the provided Python client to interact with the API:

```bash
python benchmark_api_client.py --api-url http://localhost:8080 --model-path /models/Llama-3.3-70B-Instruct/ --example both
```

## API Endpoints

### Start a Throughput Benchmark

```
POST /api/benchmark/throughput
```

Example request:

```json
{
  "model": "/models/Llama-3.3-70B-Instruct/",
  "dtype": "bfloat16",
  "tensor_parallel_size": 4,
  "pipeline_parallel_size": 1,
  "max_num_batched_tokens": 8192,
  "max_model_len": 32768,
  "max_num_seqs": 128,
  "gpu_memory_utilization": 0.90,
  "quantization": "fp8",
  "input_len": 150,
  "output_len": 200,
  "num_prompts": 1000,
  "n": 1,
  "async_engine": false
}
```

### Start a Serving Benchmark

```
POST /api/benchmark/serving
```

Example request:

```json
{
  "backend": "openai",
  "host": "localhost",
  "port": 5000,
  "endpoint": "/v1/completions",
  "model": "Llama-3.3-70B-Instruct",
  "dataset_name": "random",
  "request_rate": "inf",
  "num_prompts": 100,
  "random_input_len": 150,
  "random_output_len": 200,
  "stream": false
}
```

### Get Benchmark Result

```
GET /api/benchmark/{benchmark_id}
```

### List All Benchmarks

```
GET /api/benchmarks
```

### List Completed Benchmarks

```
GET /api/benchmarks/completed
```

### Download Benchmark Report

```
GET /api/benchmark/{benchmark_id}/report?format={format}
```

Supported formats:
- `json` (default): Returns the benchmark result in JSON format
- `csv`: Returns the benchmark result in CSV format

## Example Usage with curl

### Start a Throughput Benchmark

```bash
curl -X POST "http://localhost:8080/api/benchmark/throughput" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Llama-3.3-70B-Instruct/",
    "dtype": "bfloat16",
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1,
    "max_num_batched_tokens": 8192,
    "max_model_len": 32768,
    "max_num_seqs": 128,
    "gpu_memory_utilization": 0.90,
    "quantization": "fp8",
    "input_len": 150,
    "output_len": 200,
    "num_prompts": 1000
  }'
```

### Start a Serving Benchmark

```bash
curl -X POST "http://localhost:8080/api/benchmark/serving" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "openai",
    "host": "localhost",
    "port": 5000,
    "endpoint": "/v1/completions",
    "model": "/model/gemma-3-27b-it/",
    "dataset_name": "random",
    "num_prompts": 100,
    "random_input_len": 150,
    "random_output_len": 200
  }'
```

### Get Benchmark Result

```bash
curl "http://localhost:8080/api/benchmark/{benchmark_id}"
```

### List All Benchmarks

```bash
curl "http://localhost:8080/api/benchmarks"
```

### List Completed Benchmarks

```bash
curl "http://localhost:8080/api/benchmarks/completed"
```

### Download Benchmark Report

```bash
# Download as JSON (default)
curl "http://localhost:8080/api/benchmark/{benchmark_id}/report" -o report.json

# Download as CSV
curl "http://localhost:8080/api/benchmark/{benchmark_id}/report?format=csv" -o report.csv
```

## Understanding the Results

The benchmark results include:

### Throughput Benchmark Results

- `elapsed_time`: Total time taken to complete the benchmark (seconds)
- `num_requests`: Number of requests processed
- `total_num_tokens`: Total number of tokens processed (input + output)
- `requests_per_second`: Throughput in requests per second
- `tokens_per_second`: Throughput in tokens per second
- `output_tokens_per_second`: Throughput in output tokens per second
- `average_time_per_request`: Average time per request (seconds)

### Serving Benchmark Results

- `completed`: Number of successful requests
- `total_input`: Total number of input tokens
- `total_output`: Total number of output tokens
- `request_throughput`: Throughput in requests per second
- `output_throughput`: Throughput in output tokens per second
- `total_token_throughput`: Throughput in total tokens per second
- `mean_ttft_ms`: Mean time to first token (milliseconds)
- `median_ttft_ms`: Median time to first token (milliseconds)
- `mean_tpot_ms`: Mean time per output token (milliseconds)
- `median_tpot_ms`: Median time per output token (milliseconds)
- `mean_itl_ms`: Mean inter-token latency (milliseconds)
- `median_itl_ms`: Median inter-token latency (milliseconds)
- `mean_e2el_ms`: Mean end-to-end latency (milliseconds)
- `median_e2el_ms`: Median end-to-end latency (milliseconds)
- Various percentile metrics for ttft, tpot, itl, and e2el

## Advanced Usage

### Customizing Benchmark Parameters

You can customize all benchmark parameters by modifying the JSON request body. See the API documentation at `/docs` for all available parameters.

### Storing Results

The API server now automatically saves benchmark results to disk in the `benchmark_reports` directory (configurable via the `BENCHMARK_REPORTS_DIR` environment variable). Results are loaded from disk when the server starts, so they persist across server restarts.

You can download benchmark reports in JSON or CSV format using the `/api/benchmark/{benchmark_id}/report` endpoint.

