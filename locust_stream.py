from locust import HttpUser, task, constant_pacing, events
import time
import uuid
import json
import threading
from locust.argument_parser import LocustArgumentParser
import os

model_id = os.getenv("MODEL_ID", "vllm-gemma-3-27b-it")

stats_lock = threading.Lock()
total_input_tokens = 0
total_output_tokens = 0
num_requests = 0
test_start_time = None
shared_prompt = None
input_tokens = 150
max_tokens = 200


class OutputStats:
    def __init__(self):
        self.ttft = 0.0
        self.itl = []
        self.input_tokens = 0
        self.output_tokens = 0

class LLMUser(HttpUser):
    wait_time = constant_pacing(200)

    def on_start(self):
        global test_start_time, shared_prompt
        if test_start_time is None:
            test_start_time = time.perf_counter()

        if shared_prompt is None:
            shared_prompt = self.find_prompt_for_token_count(input_tokens)

    def find_prompt_for_token_count(self, target_tokens, max_iterations=10):
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
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1,
                "stream": False
            }

            response = self.client.post("/v1/chat/completions", json=payload, headers=headers, timeout=60)
            data = response.json()
            actual_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            print(actual_tokens)
            if actual_tokens == target_tokens:
                return prompt
            elif actual_tokens < target_tokens:
                low = mid + 1
                best_prompt = prompt
                best_tokens = actual_tokens
            else:
                high = mid - 1

        return best_prompt

    @task
    def chat_completion(self):
        global total_input_tokens, total_output_tokens, num_requests, shared_prompt

        prompt = shared_prompt
        request_id = str(uuid.uuid4())

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            "ignore_eos": True,
        }

        st = time.perf_counter()
        output = OutputStats()
        ttft = 0.0
        most_recent_timestamp = st

        try:
            with self.client.post("/v1/chat/completions", json=payload, headers=headers, stream=True, timeout=300, catch_response=True) as response:
                for chunk_bytes in response.iter_lines():
                    if not chunk_bytes or not chunk_bytes.startswith(b"data: "):
                        continue

                    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                    if chunk == "[DONE]":
                        break

                    timestamp = time.perf_counter()
                    data = json.loads(chunk)

                    if choices := data.get("choices"):
                        content = choices[0]["delta"].get("content")
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    if usage := data.get("usage"):
                        output.output_tokens = usage.get("completion_tokens")

                total_time = time.perf_counter() - st
                tpot = (total_time - ttft) / output.output_tokens if output.output_tokens else 0

                events.request.fire(
                    request_type="LLM",
                    name="T2C",
                    response_time=total_time * 1000,
                    response_length=(total_input_tokens+total_output_tokens),
                    response=response,
                    context={"request_id": request_id}
                )
                # 即時fire TTFT、TPOT
                events.request.fire(
                    request_type="LLM",
                    name="TTFT",
                    response_time=ttft * 1000,
                    response_length=input_tokens,
                    response=response,
                    context={"request_id": request_id}
                )

                events.request.fire(
                    request_type="LLM",
                    name="TPOT",
                    response_time=tpot * 1000,
                    response_length=output.output_tokens,
                    response=response,
                    context={"request_id": request_id}
                )

                # 累積 output tokens
                with stats_lock:
                    total_input_tokens += input_tokens 
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

