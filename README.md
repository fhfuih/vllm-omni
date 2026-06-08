# Optimization Experiments

**Move preprocess VAE image device to CUDA**:

- e2e latency dropped.
- Not captured by profiler (before profiler)

**Also move preprocess condition image to CUDA**:

- e2e latency recovered to before. Still no improvement
- Not captured by profiler (before profiler)
