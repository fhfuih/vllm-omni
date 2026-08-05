vllm serve . \
  --omni \
  --model-class-name DummyRealtimeVideoPipeline \
  --diffusion-streaming-output \
  --enforce-eager \
  --port 8000
