from benchmarks.demo2026.parity.weight_maps._helpers import MapEntry, WeightMap
from benchmarks.demo2026.parity.weight_maps import gpt2, llama3, qwen2

MAP_BUILDERS = {
    "gpt2": gpt2.build_map,
    "llama3": llama3.build_map,
    "qwen2": qwen2.build_map,
}

__all__ = ["MapEntry", "WeightMap", "MAP_BUILDERS"]
