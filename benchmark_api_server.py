#!/usr/bin/env python3
"""
API server for running vLLM benchmarks.
This server provides endpoints to run throughput and serving benchmarks via HTTP API.
"""
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import csv
import json
import os
import sys
import traceback
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import existing benchmark modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_throughput import run_vllm, run_vllm_async, sample_requests
from benchmark_serving import benchmark, sample_random_requests, sample_random_requests_with_engine_tokenization, sample_sharegpt_requests
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from backend_request_func import get_tokenizer

app = FastAPI(
    title="vLLM Benchmark API Server", 
    description="API server for running vLLM benchmarks",
    version="1.0.0"
)

# Configure reports storage directory
REPORTS_DIR = os.environ.get("BENCHMARK_REPORTS_DIR", "benchmark_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Dictionary to store benchmark results
benchmark_results = {}

def save_benchmark_result(benchmark_id: str, result: Dict[str, Any]) -> str:
    """
    Save benchmark result to disk
    
    Args:
        benchmark_id: ID of the benchmark
        result: Benchmark result to save
        
    Returns:
        file_path: Path to the saved file
    """
    file_path = os.path.join(REPORTS_DIR, f"{benchmark_id}.json")
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)
    return file_path

def convert_json_to_csv(json_path: str, csv_path: str) -> str:
    """
    Convert JSON benchmark result to CSV format
    
    Args:
        json_path: Path to JSON file
        csv_path: Path to save CSV file
        
    Returns:
        csv_path: Path to the saved CSV file
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract result data
    result = data.get("result", {})
    if not result:
        # If no result data, create a simple CSV with status
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "status", "start_time", "end_time"])
            writer.writerow([
                data.get("id", ""),
                data.get("status", ""),
                data.get("start_time", ""),
                data.get("end_time", "")
            ])
        return csv_path
    
    # Flatten nested dictionaries for CSV format
    flat_data = {
        "id": data.get("id", ""),
        "status": data.get("status", ""),
        "start_time": data.get("start_time", ""),
        "end_time": data.get("end_time", ""),
    }
    
    # Add all result fields
    for key, value in result.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat_data[key] = value
    
    # Write to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_data.keys())
        writer.writeheader()
        writer.writerow(flat_data)
    
    return csv_path

# Data models for API requests and responses
class ThroughputBenchmarkRequest(BaseModel):
    model: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: int = 32768
    max_num_seqs: int = 128
    gpu_memory_utilization: float = 0.90
    quantization: Optional[str] = "fp8"
    input_len: int = 150
    output_len: int = 200
    num_prompts: int = 1000
    n: int = 1
    async_engine: bool = False
    
class ServingBenchmarkRequest(BaseModel):
    backend: str = "openai"
    host: str = "localhost"
    port: int = 5000
    endpoint: str = "/v1/completions"
    model: str
    tokenizer: Optional[str] = None
    dataset_name: str = "random"
    dataset_path: Optional[str] = None
    request_rate: float = float("inf")
    num_prompts: int = 100
    random_input_len: int = 150
    random_output_len: int = 200
    stream: bool = False

class BenchmarkResult(BaseModel):
    id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Background task functions
async def run_throughput_benchmark_task(benchmark_id: str, request: ThroughputBenchmarkRequest):
    """Run throughput benchmark in the background"""
    try:
        # Convert request parameters to EngineArgs
        engine_args = EngineArgs(
            model=request.model,
            dtype=request.dtype,
            tensor_parallel_size=request.tensor_parallel_size,
            pipeline_parallel_size=request.pipeline_parallel_size,
            max_num_batched_tokens=request.max_num_batched_tokens,
            max_model_len=request.max_model_len,
            max_num_seqs=request.max_num_seqs,
            gpu_memory_utilization=request.gpu_memory_utilization,
            quantization=request.quantization
        )
        
        # Get tokenizer
        tokenizer = get_tokenizer(request.model, trust_remote_code=True)
        
        # Generate synthetic requests
        input_requests = []
        for i in range(-10, 10):
            prompt = "hi " * (request.input_len + i)
            tokenized_prompt = tokenizer(prompt).input_ids
            if len(tokenized_prompt) == request.input_len:
                break
        else:
            raise ValueError(f"Failed to synthesize a prompt with {request.input_len} tokens.")
        
        input_requests = [
            (prompt, request.input_len, request.output_len)
            for _ in range(request.num_prompts)
        ]
        
        # Run benchmark
        if request.async_engine:
            async_engine_args = AsyncEngineArgs.from_engine_args(engine_args)
            elapsed_time = await run_vllm_async(input_requests, request.n, async_engine_args)
        else:
            elapsed_time = run_vllm(input_requests, request.n, engine_args)
        
        # Calculate results
        total_num_tokens = sum(req[1] + req[2] for req in input_requests)
        total_output_tokens = sum(req[2] for req in input_requests)
        
        result = {
            "elapsed_time": elapsed_time,
            "num_requests": len(input_requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(input_requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
            "output_tokens_per_second": total_output_tokens / elapsed_time,
            "average_time_per_request": elapsed_time / len(input_requests)
        }
        
        # Update results
        benchmark_results[benchmark_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "result": result
        })
        
        # Save result to disk
        save_benchmark_result(benchmark_id, benchmark_results[benchmark_id])
        
    except Exception as e:
        # Handle errors
        exc_info = sys.exc_info()
        error_msg = "".join(traceback.format_exception(*exc_info))
        benchmark_results[benchmark_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": error_msg
        })

async def run_serving_benchmark_task(benchmark_id: str, request: ServingBenchmarkRequest):
    """Run serving benchmark in the background"""
    try:
        # Build API URL
        api_url = f"http://{request.host}:{request.port}{request.endpoint}"
        base_url = f"http://{request.host}:{request.port}"
        
        # Get tokenizer
        #tokenizer = get_tokenizer(request.model, trust_remote_code=True)
       
        # Get tokenizer
        if request.tokenizer:
            tokenizer = get_tokenizer(request.tokenizer, trust_remote_code=True)
        else:
            tokenizer = get_tokenizer(request.model, trust_remote_code=True)

        # Generate requests
        if request.dataset_name == "random":
            input_requests = sample_random_requests(
                prefix_len=0,
                input_len=request.random_input_len,
                output_len=request.random_output_len,
                num_prompts=request.num_prompts,
                range_ratio=1.0,
                tokenizer=tokenizer
            )
        elif request.dataset_name == "random_plus":
            input_requests =await sample_random_requests_with_engine_tokenization(
                prefix_len=0,
                input_len=request.random_input_len,
                output_len=request.random_output_len,
                num_prompts=request.num_prompts,
                range_ratio=1.0,
                tokenizer=tokenizer,
                backend=request.backend,
                api_url=api_url,
                model_id=request.model,
            )
        elif request.dataset_name == "sharegpt":
            if not request.dataset_path:
                raise ValueError("dataset_path is required for sharegpt dataset")
            input_requests = sample_sharegpt_requests(
                dataset_path=request.dataset_path,
                num_requests=request.num_prompts,
                tokenizer=tokenizer
            )
        else:
            raise ValueError(f"Unsupported dataset_name: {request.dataset_name}")
        
        # Run benchmark
        result = await benchmark(
            backend=request.backend,
            api_url=api_url,
            base_url=base_url,
            model_id=request.model,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=None,
            best_of=1,
            request_rate=request.request_rate,
            burstiness=1.0,
            disable_tqdm=True,
            profile=False,
            selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
            selected_percentiles=[50, 90, 95, 99],
            ignore_eos=True,
            gootput_config_dict={},
            max_concurrency=None,
            stream=request.stream
        )
        
        # Update results
        benchmark_results[benchmark_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "result": result
        })
        
    except Exception as e:
        # Handle errors
        exc_info = sys.exc_info()
        error_msg = "".join(traceback.format_exception(*exc_info))
        benchmark_results[benchmark_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": error_msg
        })

# API endpoints
@app.post("/api/benchmark/throughput", response_model=BenchmarkResult)
async def start_throughput_benchmark(request: ThroughputBenchmarkRequest, background_tasks: BackgroundTasks):
    """Start a throughput benchmark"""
    benchmark_id = f"throughput_{datetime.now().strftime('%Y%m%d%H%M%S')}_{request.model.split('/')[-1]}"
    
    # Initialize result
    benchmark_results[benchmark_id] = {
        "id": benchmark_id,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    # Run benchmark in background
    background_tasks.add_task(
        run_throughput_benchmark_task,
        benchmark_id=benchmark_id,
        request=request
    )
    
    return BenchmarkResult(
        id=benchmark_id,
        status="running",
        start_time=benchmark_results[benchmark_id]["start_time"]
    )

@app.post("/api/benchmark/serving", response_model=BenchmarkResult)
async def start_serving_benchmark(request: ServingBenchmarkRequest, background_tasks: BackgroundTasks):
    """Start a serving benchmark"""
    benchmark_id = f"serving_{datetime.now().strftime('%Y%m%d%H%M%S')}_{request.model.split('/')[-1]}"
    
    # Initialize result
    benchmark_results[benchmark_id] = {
        "id": benchmark_id,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    # Run benchmark in background
    background_tasks.add_task(
        run_serving_benchmark_task,
        benchmark_id=benchmark_id,
        request=request
    )
    
    return BenchmarkResult(
        id=benchmark_id,
        status="running",
        start_time=benchmark_results[benchmark_id]["start_time"]
    )

@app.get("/api/benchmark/{benchmark_id}", response_model=BenchmarkResult)
async def get_benchmark_result(benchmark_id: str):
    """Get benchmark result by ID"""
    if benchmark_id not in benchmark_results:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    result = benchmark_results[benchmark_id]
    return BenchmarkResult(
        id=result["id"],
        status=result["status"],
        start_time=result["start_time"],
        end_time=result.get("end_time"),
        result=result.get("result"),
        error=result.get("error")
    )

@app.get("/api/benchmarks", response_model=List[BenchmarkResult])
async def list_benchmarks():
    """List all benchmarks"""
    return [
        BenchmarkResult(
            id=result["id"],
            status=result["status"],
            start_time=result["start_time"],
            end_time=result.get("end_time"),
            result=None,  # Don't return full results to reduce response size
            error=result.get("error")
        )
        for result in benchmark_results.values()
    ]

@app.get("/api/benchmarks/completed", response_model=List[BenchmarkResult])
async def list_completed_benchmarks():
    """List all completed benchmarks"""
    return [
        BenchmarkResult(
            id=result["id"],
            status=result["status"],
            start_time=result["start_time"],
            end_time=result.get("end_time"),
            result=None,  # Don't return full results to reduce response size
            error=result.get("error")
        )
        for result in benchmark_results.values()
        if result["status"] == "completed"
    ]

@app.get("/api/benchmark/{benchmark_id}/report")
async def get_benchmark_report(benchmark_id: str, format: str = Query("json", regex="^(json|csv)$")):
    """
    Get benchmark report in specified format
    
    Args:
        benchmark_id: ID of the benchmark
        format: Format of the report (json or csv)
    """
    if benchmark_id not in benchmark_results:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    result = benchmark_results[benchmark_id]
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail="Benchmark not completed")
    
    # Check if report file exists
    json_path = os.path.join(REPORTS_DIR, f"{benchmark_id}.json")
    if not os.path.exists(json_path):
        # If file doesn't exist, create it
        save_benchmark_result(benchmark_id, result)
    
    if format == "json":
        return FileResponse(
            path=json_path,
            filename=f"{benchmark_id}.json",
            media_type="application/json"
        )
    elif format == "csv":
        # Convert JSON to CSV
        csv_path = os.path.join(REPORTS_DIR, f"{benchmark_id}.csv")
        convert_json_to_csv(json_path, csv_path)
        return FileResponse(
            path=csv_path,
            filename=f"{benchmark_id}.csv",
            media_type="text/csv"
        )

@app.on_event("startup")
async def startup_event():
    """Load saved benchmark results on startup"""
    if os.path.exists(REPORTS_DIR):
        for filename in os.listdir(REPORTS_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(REPORTS_DIR, filename)
                try:
                    with open(file_path, "r") as f:
                        result = json.load(f)
                        benchmark_id = result["id"]
                        benchmark_results[benchmark_id] = result
                        print(f"Loaded benchmark result: {benchmark_id}")
                except Exception as e:
                    print(f"Error loading benchmark result from {file_path}: {e}")

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "name": "vLLM Benchmark API Server",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "/api/benchmark/throughput",
            "/api/benchmark/serving",
            "/api/benchmark/{benchmark_id}",
            "/api/benchmark/{benchmark_id}/report",
            "/api/benchmarks",
            "/api/benchmarks/completed"
        ]
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Benchmark API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    
    args = parser.parse_args()
    
    print(f"Starting vLLM Benchmark API Server on {args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)

