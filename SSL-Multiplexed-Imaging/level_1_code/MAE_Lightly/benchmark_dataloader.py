import time
from pathlib import Path
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset

import chunked_h5_dataset
from mae_custom_transform import MAETransform


def benchmark_dataloader(dataloader, n_batches=200, warmup_batches=20, move_to_gpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    it = iter(dataloader)

    # warmup
    t0 = time.perf_counter()
    for w in range(warmup_batches):
        batch = next(it)
        if move_to_gpu:
            views = batch[0]
            x = views[0]
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

        if (w + 1) % 1 == 0:  # print every warmup batch
            elapsed = time.perf_counter() - t0
            print(f"Warmup {w+1}/{warmup_batches}  elapsed={elapsed:.2f}s", flush=True)

    t_warm = time.perf_counter() - t0

    # timed
    t0 = time.perf_counter()
    n_samples = 0
    for b in range(n_batches):
        batch = next(it)
        views = batch[0]
        x = views[0]
        n_samples += x.shape[0]

        if move_to_gpu:
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

        if (b + 1) % 10 == 0:  # print every 10 batches
            elapsed = time.perf_counter() - t0
            rate = n_samples / elapsed if elapsed > 0 else float("inf")
            print(f"[{b+1}/{n_batches}] elapsed={elapsed:.2f}s  rate={rate:.1f} samples/s", flush=True)
    

    t = time.perf_counter() - t0
    sec_per_batch = t / n_batches
    samples_per_sec = n_samples / t

    print("\n=== DataLoader Benchmark ===")
    print(f"Warmup: {warmup_batches} batches -> {t_warm:.3f}s")
    print(f"Timed:  {n_batches} batches -> {t:.3f}s")
    print(f"Avg:    {sec_per_batch:.4f} s/batch")
    print(f"Rate:   {samples_per_sec:.1f} samples/s")
    print(f"move_to_gpu={move_to_gpu}, device={device}")


def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    pl.seed_everything(args.seed)

    mean_std, _ = chunked_h5_dataset.get_mean_std()
    transform = MAETransform(input_size=args.input_size, min_scale=0.2, normalize=mean_std)

    base = chunked_h5_dataset.h5_chunk_wrapper(Path(args.data_path))
    dataset = LightlyDataset.from_torch_dataset(base, transform=transform)
    print("Loaded dataset length:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    benchmark_dataloader(
        dataloader,
        n_batches=args.n_batches,
        warmup_batches=args.warmup_batches,
        move_to_gpu=args.move_to_gpu,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser("HDF5 DataLoader benchmark (no training)")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup_batches", type=int, default=20)
    p.add_argument("--n_batches", type=int, default=200)
    p.add_argument("--move_to_gpu", action="store_true")
    args = p.parse_args()
    main(args)