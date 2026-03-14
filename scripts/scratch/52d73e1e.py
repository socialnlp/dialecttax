"""Load a model on each given GPU and hammer it with random forward passes
in parallel to keep utilization >97%. Silent — no printing, no progress
counters. One process per GPU (multiprocessing, not threading — threads
serialize on the GIL and cap utilization in the low teens). Pass
--model_ids in the same order as --gpus, or a single value to broadcast.

Usage:
    python scripts/scratch/52d73e1e.py
    python scripts/scratch/52d73e1e.py --model_ids Qwen/Qwen3-8B --gpus 0 6
    python scripts/scratch/52d73e1e.py \\
        --gpus 0 1 2 3 4 5 6 7 \\
        --model_ids Qwen/Qwen3-8B
"""

import argparse
import os

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM


def worker(gpu, model_id, batch_size, seq_len):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    g = torch.Generator(device=device).manual_seed(gpu)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len),
        generator=g, device=device, dtype=torch.long,
    )
    while True:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument(
        "--model_ids", nargs="+",
        default=["Qwen/Qwen3-8B"] * 8,
    )
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if len(args.model_ids) == 1:
        model_ids = args.model_ids * len(args.gpus)
    elif len(args.model_ids) == len(args.gpus):
        model_ids = args.model_ids
    else:
        parser.error(
            f"--model_ids must have length 1 or len(--gpus)={len(args.gpus)}, "
            f"got {len(args.model_ids)}"
        )

    ctx = mp.get_context("spawn")
    procs = []
    for gpu, model_id in zip(args.gpus, model_ids):
        p = ctx.Process(
            target=worker,
            args=(gpu, model_id, args.batch_size, args.seq_len),
            daemon=False,
        )
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=10)


if __name__ == "__main__":
    main()
