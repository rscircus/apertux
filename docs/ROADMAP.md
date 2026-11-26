# Roadmap

## Next

- Local `FP8_DYNAMIC` / `NVFP4` quantization (`llm-compressor`) paired with speculative decoding (`Apertus-v1.1-1.5B-Instruct` draft or `ngram`) to reach ~20+ tok/s interactive decode on 2x DGX Spark.
- Dual-port ConnectX-7 400GbE bonding (`bond0`) option to halve inter-node all-reduce latency.
