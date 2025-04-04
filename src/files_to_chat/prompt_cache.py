import json
import sys
import time

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache
from mlx_lm.utils import load
from mlx_lm.generate import generate_step


def create_prompt_cache(model_path: str, prompt: str, cache_path: str):
    model, tokenizer = load(model_path)
    prompt = tokenizer.encode(prompt)
    cache = make_prompt_cache(model)
    y = mx.array(prompt)

    start = time.time()
    max_msg_len = 0

    def callback(processed, total_tokens):
        current = time.time()
        speed = processed / (current - start)
        msg = f"\rProcessed {processed:6d} tokens ({speed:6.2f} tok/s)"
        nonlocal max_msg_len
        max_msg_len = max(max_msg_len, len(msg))
        print(msg + " " * (max_msg_len - len(msg)), end="", flush=True)

    for _ in generate_step(
        y,
        model,
        max_tokens=0,
        prompt_cache=cache,
        prompt_progress_callback=callback,
    ):
        pass

    print()
    print(f"Peak memory: {mx.metal.get_peak_memory() / 1e9:.3f} GB")

    print("Saving...")
    metadata = {}
    metadata["model"] = model_path
    metadata["chat_template"] = tokenizer.chat_template
    metadata["tokenizer_config"] = json.dumps({})
    save_prompt_cache(cache_path, cache, metadata)
    return cache, model, tokenizer
