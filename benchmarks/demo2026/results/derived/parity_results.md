# Numerical parity (OLM vs Hugging Face Transformers)

Tiny FP32 configs; dropout off; deterministic CPU. Tolerances were **not**
chosen in advance — values below are measured errors.

| family | seed | max |dlogit| | mean |dlogit| | |dloss| | cos(emb) | cos(early) | cos(late) | status |
|--------|------|-------------|--------------|---------|----------|------------|-----------|--------|
| gpt2 | 11 | 3.129e-07 | 5.063e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| gpt2 | 22 | 3.129e-07 | 4.900e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| gpt2 | 33 | 2.980e-07 | 4.954e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| llama3 | 11 | 2.384e-07 | 3.443e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| llama3 | 22 | 2.384e-07 | 3.508e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| llama3 | 33 | 2.682e-07 | 3.552e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| qwen2 | 11 | 3.576e-07 | 3.571e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| qwen2 | 22 | 3.576e-07 | 3.352e-08 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 | complete |
| qwen2 | 33 | 2.980e-07 | 3.527e-08 | 4.768e-07 | 1.000000 | 1.000000 | 1.000000 | complete |
