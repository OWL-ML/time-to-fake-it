import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    ## add check for datasets that return none samples for missing items
    if sample is None or len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class PrefetchLoader(object):
    """
    Modified from https://github.com/ChenRocks/UNITER.

    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None and len(batch) > 0:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __next__(self, it):
        return self.next(it)

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


def seed_worker(worker_id):
    """Seed NumPy and Python random in each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(dataset, cfg, is_trains=True, collate_fns=None):
    """Create a seeded DataLoader, wrapped in a PrefetchLoader on CUDA."""
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", cfg.get("base_seed", 0)))

    # Generator tied to the run seed for reproducible worker RNG.
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=bool(is_trains),
        collate_fn=collate_fns,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # CUDA prefetch only when CUDA is available.
    if torch.cuda.is_available():
        return PrefetchLoader(loader)
    return loader
