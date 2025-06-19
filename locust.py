from locust import HttpUser, task, constant_pacing, events
import time
import uuid
import json
import threading
import os
import requests

stats_lock = threading.Lock()
total_input_tokens = 0
total_output_tokens = 0
num_requests = 0
test_start_time = None
shared_prompt = None
prompt_ready = threading.Event()
input_tokens = 150
max_tokens = 200

class OutputStats:
    def __init__(self):
        self.ttft = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

def find_prompt_for_token_count(client_post, target_tokens, model, max_iterations=10):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }

    base_token = "<a>"
    low = 1
    high = target_tokens * 2
    best_prompt = base_token
    best_tokens = 0

    for _ in range(max_iterations):
        mid = (low + high) // 2
        prompt = base_token * mid

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1,
            "stream": False
        }

        response = client_post("/v1/chat/completions", json=payload, headers=headers, timeout=60)
        data = response.json()
        actual_tokens = data.get("usage", {}).get("prompt_tokens", 0)

        if actual_tokens == target_tokens:
            return prompt
        elif actual_tokens < target_tokens:
            low = mid + 1
            best_prompt = prompt
            best_tokens = actual_tokens
        else:
            high = mid - 1

    return best_prompt

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    global shared_prompt

    print("Initializing shared prompt...")
    model = os.getenv("MODEL_ID", "vllm-gemma-3-27b-it")

    def client_post(path, **kwargs):
        return requests.post(f"{environment.host}{path}", **kwargs)

    shared_prompt = find_prompt_for_token_count(client_post, input_tokens, model)
    prompt_ready.set()
    print("Shared prompt initialized.")

class LLMUser(HttpUser):
    wait_time = constant_pacing(200)

    def on_start(self):
        global test_start_time
        if test_start_time is None:
            test_start_time = time.perf_counter()

        prompt_ready.wait()

    @task
    def chat_completion(self):
        global total_input_tokens, total_output_tokens, num_requests, shared_prompt

        prompt = shared_prompt
        request_id = str(uuid.uuid4())

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"
        }

        model = os.getenv("MODEL_ID", "vllm-gemma-3-27b-it")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "ignore_eos": True,

        }

        st = time.perf_counter()
        output = OutputStats()

        try:
            with self.client.post("/v1/chat/completions", json=payload, headers=headers, timeout=300, catch_response=True) as response:
                data = response.json()
                timestamp = time.perf_counter()

                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    if content:
                        output.ttft = timestamp - st

                if "usage" in data:
                    output.input_tokens = data["usage"].get("prompt_tokens", 0)
                    output.output_tokens = data["usage"].get("completion_tokens", 0)

                total_time = time.perf_counter() - st

                events.request.fire(
                    request_type="LLM",
                    name="T2C",
                    response_time=total_time * 1000,
                    response_length=output.input_tokens + output.output_tokens,
                    response=response,
                    context={"request_id": request_id}
                )

                events.request.fire(
                    request_type="LLM",
                    name="TTFT",
                    response_time=output.ttft * 1000,
                    response_length=output.input_tokens,
                    response=response,
                    context={"request_id": request_id}
                )

                with stats_lock:
                    total_input_tokens += output.input_tokens
                    total_output_tokens += output.output_tokens
                    num_requests += 1

                response.success()

        except Exception as e:
            events.request.fire(
                request_type="LLM",
                name="LLM Request Failed",
                response_time=(time.perf_counter() - st) * 1000,
                response_length=0,
                exception=e,
                context={"request_id": request_id}
            )

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    global test_start_time
    test_end_time = time.perf_counter()
    total_time = test_end_time - test_start_time if test_start_time else 1

    with stats_lock:
        if num_requests > 0:
            avg_tokens_per_sec = (total_input_tokens + total_output_tokens) / total_time if total_time > 0 else 0
            print(f"Test finished. Total requests: {num_requests}")
            print(f"Total input tokens: {total_input_tokens}")
            print(f"Total output tokens: {total_output_tokens}")
            print(f"Test total time: {total_time:.2f} seconds")
            print(f"Average throughput (tokens/sec): {avg_tokens_per_sec:.2f}")
        else:
            print("No successful requests recorded.")

