#!/bin/bash
# Start the vLLM Benchmark API Server

# Default values
HOST="0.0.0.0"
PORT="8080"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--host HOST] [--port PORT]"
      exit 1
      ;;
  esac
done

# Make the script executable
chmod +x benchmark_api_server.py

# Start the server
echo "Starting vLLM Benchmark API Server on $HOST:$PORT"
python3 benchmark_api_server.py --host "$HOST" --port "$PORT"

