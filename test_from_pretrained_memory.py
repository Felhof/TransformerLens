import argparse
import json

from memory_profiler import memory_usage

from transformer_lens import HookedTransformer


def load_model(model_name, load_in_chunks=False):
    HookedTransformer.from_pretrained(
        model_name,
        device="cuda",
        load_in_chunks=load_in_chunks,
    )


def profile_loading(*args, **kwargs):
    mem_usage, result = memory_usage((load_model, args, kwargs), retval=True)
    return mem_usage, result


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, help="The model for which to profile memory usage")
parser.add_argument("--chunkwise", action="store_true", help="Enable chunkwise processing")

args = parser.parse_args()

mem_usage_before_loading = memory_usage(-1)


mem_usage, _ = profile_loading(
    args.model_name,
    load_in_chunks=args.chunkwise,
)

mem_usage = [mem - mem_usage_before_loading[0] for mem in mem_usage]

with open("mem_usage.json", "w") as f:
    json.dump(mem_usage, f)
