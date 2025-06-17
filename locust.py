from locust import HttpUser, task, constant_pacing, events
import time
import uuid
import json
import threading

stats_lock = threading.Lock()
total_input_tokens = 0
total_output_tokens = 0
num_requests = 0
test_start_time = None

class OutputStats:
    def __init__(self):
        self.ttft = 0.0
        self.itl = []
        self.output_tokens = 0

class LLMUser(HttpUser):
    wait_time = constant_pacing(200)

    def on_start(self):
        global test_start_time
        if test_start_time is None:
            test_start_time = time.perf_counter()

    @task
    def chat_completion(self):
        global total_input_tokens, total_output_tokens, num_requests

        #prompt = "Explain the theory of relativity in simple terms."
        prompt = "<a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a>"
        input_tokens = 150
        max_tokens = 200
        request_id = str(uuid.uuid4())

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"
        }

        payload = {
            "model": "vllm-gemma-3-27b-it",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
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

                # 即時fire TTFT、TPOT
                events.request.fire(
                    request_type="LLM",
                    name="TTFT",
                    response_time=ttft * 1000,
                    response_length=0,
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
            avg_tokens_per_sec = (total_input_tokens+total_output_tokens) / total_time if total_time > 0 else 0
            print(f"Test finished. Total requests: {num_requests}")
            print(f"Total input tokens: {total_input_tokens}")
            print(f"Total output tokens: {total_output_tokens}")
            print(f"Test total time: {total_time:.2f} seconds")
            print(f"Average throughput (tokens/sec): {avg_tokens_per_sec:.2f}")
        else:
            print("No successful requests recorded.")

