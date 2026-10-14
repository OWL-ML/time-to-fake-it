#!/usr/bin/env python
"""fisher_evolution_pipeline.py — curvature consistency sweep, split by
demographic group.

For EWC checkpoints, computes the empirical Fisher diagonal on each
demographic subgroup of a task's training data and compares it with the
Fisher of the whole task (cosine similarity, L1 distance, symmetrized KL).

Images without a demographic annotation (code < 0 in the demographics CSV,
e.g. -1) are not grouped: they get no subgroup of their own and are not counted
in any subgroup. --keep-missing restores the old behaviour. The reference
Fisher of a dataset still uses all of its images.

Run it with ../../run_fisher.sh (one job per seed, logs under logs/fisher/):

    ./run_fisher.sh                                   # all datasets, local reference
    SEEDS="42 123" ./run_fisher.sh fisher_sim_last_all_datasets_global
    ./run_fisher.sh single_dataset_over_time

Seeds can also run as separate jobs into the same --out-dir; the per-seed
CSVs are merged once all N_SEEDS_FOR_MERGE of them exist.

Only some datasets (e.g. re-run after fixing their annotations): write to a
separate --out-dir so the full per-seed CSVs are not overwritten, and let the
last seed replace those datasets in the existing merged CSV (a backup
<name>_before_merge.csv is written first):

    DATASETS="VQGAN Palette DCFACe" OUT_DIR=results/consistency_fix \
    MERGE_INTO=results/consistency/empirical_fisher_local_distances_merged.csv \
    ./run_fisher.sh
"""

import argparse
import gc
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import IndxImageFolder
from utils.constants import DEMOGRAPHICS, METADATA_CSV, TASK_NAMES

CHECKPOINT_DIR = Path("checkpoints/ewc")
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "datasets/ai_face"))
NUM_WORKERS = 12
N_SEEDS_FOR_MERGE = 5  # once this many per-seed CSVs exist in out-dir, merge them

transform_test = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# --------------------------------------------------------------------------- #
# Model / checkpoint helpers
# --------------------------------------------------------------------------- #


def build_model(num_classes=2, arch="vit_base_patch16_224"):
    return timm.create_model(arch, pretrained=False, num_classes=num_classes)


def load_checkpoint_model(model, checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Empirical Fisher + comparison metrics
# --------------------------------------------------------------------------- #


def compute_empirical_fisher(
    model: nn.Module,
    loss_func: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cuda",
    use_amp: bool = True,
) -> tuple[dict[str, torch.Tensor], int]:
    """
    Single-pass computation for Accuracy, Predictive Entropy, and Empirical Fisher.
    Keeps all Fisher diagonals on GPU in torch.float32 for maximum precision.
    """
    model.eval()
    model.to(device)

    # Pre-filter trainable parameters to avoid string dict lookups inside batch loop
    trainable_params = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]

    # Keep accumulator directly on CUDA device
    fisher_diag = {
        name: torch.zeros_like(param, dtype=torch.float32, device=device)
        for name, param in trainable_params
    }

    total_samples = 0

    for x, y, *_ in loader:  # IndxImageFolder also yields the image path
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.size(0)
        total_samples += batch_size

        model.zero_grad(set_to_none=True)

        # Mixed precision forward/backward pass
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(x)
            loss = loss_func(outputs, y)
            if loss.ndim > 0:
                loss = loss.mean()

        loss.backward()

        # Accumulate squared gradients directly on GPU
        with torch.no_grad():
            for name, param in trainable_params:
                if param.grad is not None:
                    fisher_diag[name] += (param.grad.float() * batch_size) ** 2

    # Normalize Fisher by dataset size
    with torch.no_grad():
        for name in fisher_diag:
            fisher_diag[name] /= max(total_samples, 1)

    model.zero_grad(set_to_none=True)
    return fisher_diag, total_samples


def compute_gpu_diagonal_distances(
    sub_diag: dict[str, torch.Tensor],
    ref_diag: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Computes Cosine Similarity, L1 Distance, and Symmetrized KL Divergence
    entirely on GPU memory to prevent CPU thread starvation.
    """
    with torch.no_grad():
        # Flatten parameter vectors directly on CUDA
        sub_flat = torch.cat([sub_diag[k].flatten() for k in sub_diag])
        ref_flat = torch.cat([ref_diag[k].flatten() for k in ref_diag])

        # 1. Cosine Similarity
        sub_norm = torch.linalg.norm(sub_flat)
        ref_norm = torch.linalg.norm(ref_flat)
        cosine_similarity = (
            torch.dot(sub_flat, ref_flat) / max(sub_norm * ref_norm, eps)
        ).item()

        # 2. Normalized L1 Distance
        l1_distance = torch.mean(torch.abs(sub_flat - ref_flat)).item()

        # 3. Symmetrized KL Divergence
        p = sub_flat / (sub_flat.sum() + eps)
        q = ref_flat / (ref_flat.sum() + eps)
        symmetrized_kl = (
            0.5
            * torch.sum(
                p * torch.log((p + eps) / (q + eps))
                + q * torch.log((q + eps) / (p + eps))
            )
        ).item()

    return {
        "cosine_similarity": cosine_similarity,
        "l1_distance": l1_distance,
        "symmetrized_kl": symmetrized_kl,
    }


# --------------------------------------------------------------------------- #
# Demographic grouping
# --------------------------------------------------------------------------- #


def demo_df(dataset) -> pd.DataFrame:
    """dataset.demographics is {attr: np.ndarray}, positionally parallel to
    dataset.samples. Convert to DataFrame with matching integer index so
    .groupby(attr).groups values are valid indices into the dataset/Subset."""
    d = dataset.demographics
    df = d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)
    df.index = range(len(df))
    return df


def is_missing_code(code) -> bool:
    """True for codes that mark a missing annotation: NaN/None, or a number < 0
    (the demographics CSV uses -1 for images without annotation)."""
    if code is None or pd.isna(code):
        return True
    try:
        return float(code) < 0
    except (TypeError, ValueError):
        return False


def build_demographic_groups_single_dataset(
    dataset, demo_attrs, max_per_group=None, seed=0, skip_missing=True
):
    """Splits ONE dataset by demographic group (attr, code) -> Subset.

    No cross-source pooling here -- that's PooledDemographicDataset's job for
    the multi-dataset case. This is for the "load the first dataset, split by
    demographics" workflow.

    max_per_group: optional cap per (attr, code) group, sampled deterministically
    via `seed`. None = use every sample with a non-null label for that attr.
    skip_missing: leave out codes < 0 (missing annotation), so images without
    annotation form no group of their own. False: only NaN is left out.
    """
    df = demo_df(dataset)
    rng = random.Random(seed)
    groups = {}
    for attr in demo_attrs:
        if attr not in df.columns:
            continue
        for code, idx_index in df.groupby(attr).groups.items():
            if pd.isna(code) or (skip_missing and is_missing_code(code)):
                continue
            idxs = list(idx_index)
            if max_per_group is not None:
                rng.shuffle(idxs)
                idxs = idxs[:max_per_group]
            groups[(attr, str(code))] = Subset(dataset, idxs)
    return groups


def build_data_train(
    dataset_dir: str | Path,
    demo_csv: str | Path | None = None,
    task_order: tuple = tuple(TASK_NAMES),
    demo_attrs: tuple = tuple(DEMOGRAPHICS),
) -> dict:
    """
    dataset_dir: root containing one subfolder per task/generator (e.g.
        <dataset_dir>/ff++/train/...).
    demo_csv: path to the demographics CSV (has a path column). Defaults to
        the training-split annotations, METADATA_CSV["train"].
    """

    dataset_dir = Path(dataset_dir)
    demo_csv = Path(demo_csv) if demo_csv is not None else Path(METADATA_CSV["train"])
    assert demo_csv.exists(), f"missing {demo_csv}"

    data_train = {}
    for generator_model in task_order:
        testset = IndxImageFolder(
            root=os.path.join(dataset_dir, generator_model, "train"),
            transform=transform_test,
            num_classes=2,
            return_demographics=False,
        )
        testset.attach_demographics(str(demo_csv), demo_attrs=list(demo_attrs))
        data_train[generator_model] = testset
    return data_train


# --------------------------------------------------------------------------- #
# Result writing / merging
# --------------------------------------------------------------------------- #


def _seed_results_to_rows(seed, seed_dict: dict) -> list[dict]:
    rows = []
    for key, group_dict in seed_dict.items():
        # Handle 3-tuple (dataset, attr, code) vs 2-tuple (attr, code)
        if len(key) == 3:
            dataset_name, attr, code = key
        else:
            dataset_name = None
            attr, code = key

        for ckpt_id, metrics in group_dict.items():
            row = {"seed": seed}
            if dataset_name is not None:
                row["dataset"] = dataset_name
            row.update(
                {
                    "attr": attr,
                    "code": code,
                    "ckpt_id": ckpt_id,
                    **metrics,
                }
            )
            rows.append(row)
    return rows


def get_csv_prefix(experiment_name: str) -> str:
    qualifier = "global" if "global" in experiment_name else "local"
    prefix = f"empirical_fisher_{qualifier}_distances"
    if "single" in experiment_name:
        prefix = f"single_dataset_{prefix}"
    return prefix


def write_seed_results(seed, seed_dict: dict, out_dir: Path, experiment_name) -> Path:
    """Writes one seed's results to its own CSV (seeds are often run as
    separate, split-up jobs -- keeping them in separate files avoids
    cross-job write contention)."""
    df = pd.DataFrame(_seed_results_to_rows(seed, seed_dict))
    prefix = get_csv_prefix(experiment_name)
    path = out_dir / f"{prefix}_seed_{seed}.csv"
    df.to_csv(path, index=False)
    return path


def maybe_merge_seed_csvs(
    out_dir: Path, experiment_name, expected_n_seeds: int = N_SEEDS_FOR_MERGE
) -> Path | None:
    """If exactly `expected_n_seeds` per-seed CSVs are present in out_dir
    (across however many jobs produced them), merge them into one CSV."""
    prefix = get_csv_prefix(experiment_name)
    seed_csvs = sorted(
        [
            p
            for p in out_dir.glob(f"{prefix}_seed_*.csv")
            if not p.name.endswith("merged.csv")
        ]
    )
    if len(seed_csvs) != expected_n_seeds:
        logging.info(
            f"{len(seed_csvs)}/{expected_n_seeds} seed csv(s) present in {out_dir}, not merging yet"
        )
        return None

    merged = pd.concat([pd.read_csv(p) for p in seed_csvs], ignore_index=True)  # noqa
    merged_path = out_dir / f"{prefix}_merged.csv"
    merged.to_csv(merged_path, index=False)
    logging.info(
        f"merged {len(seed_csvs)} seed csv(s) -> {merged_path} ({len(merged)} rows)"
    )
    return merged_path


def merge_into_existing(new_path: Path, target: Path) -> None:
    """Replace the datasets of new_path in target (another merged CSV), keeping
    all other datasets. A backup <target>_before_merge.csv is written first (only
    once: a later merge, e.g. by a second job finishing, keeps the original)."""
    new = pd.read_csv(new_path)
    if not target.exists():
        logging.warning(f"--merge-into target {target} not found; nothing replaced")
        return
    old = pd.read_csv(target)
    if "dataset" not in new.columns or "dataset" not in old.columns:
        logging.warning(
            "--merge-into needs a 'dataset' column in both CSVs; nothing replaced"
        )
        return
    backup = target.with_name(f"{target.stem}_before_merge.csv")
    if not backup.exists():  # keep the very first original if merged more than once
        old.to_csv(backup, index=False)
    replaced = sorted(new["dataset"].unique())
    keep = old[~old["dataset"].isin(replaced)]
    out = pd.concat([keep, new], ignore_index=True)
    out.to_csv(target, index=False)
    logging.info(
        f"replaced datasets {replaced} in {target}: {len(old)} -> {len(out)} rows "
        f"(backup: {backup})"
    )


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #


def run_single_dataset_over_time_experiment(args) -> dict:
    """
    The "evaluate one dataset over time" pipeline: load a single dataset,
    split it by demographic group, then for every (seed, checkpoint) compute
    the empirical Fisher diagonal on the full dataset (reference) and on each
    demographic subgroup, and compare subgroup vs. reference via cosine
    similarity / L1 distance / symmetrized KL.

    Writes one CSV per seed as soon as that seed is done, then merges once
    N_SEEDS_FOR_MERGE per-seed CSVs exist in --out-dir.
    """
    out_dir = Path(args.out_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_name = args.dataset_name or TASK_NAMES[0]

    logging.info(f"loading single dataset '{dataset_name}' (train split)...")
    data_train = build_data_train(
        dataset_dir=args.dataset_dir,
        demo_csv=args.demo_csv,
        task_order=tuple([dataset_name]),
    )
    dataset = data_train[dataset_name]
    logging.info(f"  {dataset_name}: {len(dataset)} samples")

    loss_func = nn.CrossEntropyLoss()

    groups = build_demographic_groups_single_dataset(
        dataset, DEMOGRAPHICS, None, skip_missing=not args.keep_missing
    )
    logging.info(
        f"built {len(groups)} demographic groups: {[(k, len(v)) for k, v in groups.items()]}"
    )

    entire_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    results = {}

    for seed in args.seeds:
        seed_results = defaultdict(dict)

        for ckpt_id in args.checkpoints:
            ckpt_path = checkpoint_dir / f"seed_run_{seed}" / f"ewc_{ckpt_id}.pt"
            if not ckpt_path.exists():
                logging.warning(
                    f"[seed {seed}] ckpt {ckpt_id}: missing at {ckpt_path}, skipping"
                )
                continue

            model = build_model()
            model = load_checkpoint_model(model, ckpt_path)

            logging.info(
                f"[seed {seed}] ckpt {ckpt_id}: computing reference diagonal on full dataset..."
            )
            reference_empirical_diag, _ = compute_empirical_fisher(
                model, loss_func, entire_loader
            )

            for key, sub_ds in tqdm(
                list(groups.items()),
                desc=f"[seed {seed}] ckpt {ckpt_id} subgroups",
                leave=False,
            ):
                if len(sub_ds) == 0:
                    continue

                loader = DataLoader(
                    sub_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                )

                sub_diag, n = compute_empirical_fisher(model, loss_func, loader)
                metrics = compute_gpu_diagonal_distances(
                    sub_diag, reference_empirical_diag
                )

                # Immediate explicit cleanup of CUDA memory
                del sub_diag

                attr, code = key
                seed_results[(dataset_name, attr, str(code))][ckpt_id] = dict(
                    cosine_similarity=metrics["cosine_similarity"],
                    l1_distance=metrics["l1_distance"],
                    symmetrized_kl=metrics["symmetrized_kl"],
                    n=n,
                )

            # Cleanup references per checkpoint iteration
            del reference_empirical_diag
            free_model(model)
            gc.collect()
            torch.cuda.empty_cache()
            logging.info(
                f"[seed {seed}] ckpt {ckpt_id}: {len(groups)} groups processed"
            )

        results[seed] = seed_results
        seed_path = write_seed_results(
            seed, seed_results, out_dir, args.experiment_name
        )
        logging.info(f"[seed {seed}] written: {seed_path}")

    maybe_merge_seed_csvs(out_dir, args.experiment_name)

    return results


def fisher_sim_last_all_datasets(args) -> dict:
    """
    Evaluates all datasets at the final checkpoint (ckpt 11).
    Computes reference Fisher diagonal on the entire last dataset (StableDiffusion),
    then iterates over all datasets in TASK_NAMES, computing distances for each subgroup
    using all available samples per subgroup.
    """
    out_dir = Path(args.out_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_dir = Path(args.dataset_dir)

    ref_ckpt = 11
    ref_ds_name = "StableDiffusion"

    # --datasets: only these (in TASK_NAMES); default all
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(TASK_NAMES))
        if unknown:
            raise ValueError(f"--datasets not in TASK_NAMES: {unknown}")
    tasks = tuple(d for d in TASK_NAMES if not args.datasets or d in args.datasets)
    to_load = tasks
    if "global" in args.experiment_name and ref_ds_name not in to_load:
        to_load = tasks + (ref_ds_name,)  # the global reference needs it

    logging.info(f"Loading datasets {list(to_load)} for '{args.experiment_name}'...")
    data_train = build_data_train(
        dataset_dir=dataset_dir,
        demo_csv=args.demo_csv,
        task_order=to_load,
    )

    loss_func = nn.CrossEntropyLoss()
    results = {}

    for seed in args.seeds:
        seed_results = defaultdict(dict)
        ckpt_path = checkpoint_dir / f"seed_run_{seed}" / f"ewc_{ref_ckpt}.pt"
        if not ckpt_path.exists():
            logging.warning(
                f"[seed {seed}] ckpt {ref_ckpt}: missing at {ckpt_path}, skipping"
            )
            continue

        model = build_model()
        model = load_checkpoint_model(model, ckpt_path)
        reference_empirical_diag = {"": torch.zeros(1)}  # dummy

        if "global" in args.experiment_name:
            logging.info(
                f"[seed {seed}] ckpt {ref_ckpt}: computing reference diagonal on full reference dataset '{ref_ds_name}'..."
            )
            ref_dataset = data_train[ref_ds_name]
            ref_loader = DataLoader(
                ref_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )
            reference_empirical_diag, _ = compute_empirical_fisher(
                model, loss_func, ref_loader
            )

        for ds_name in tasks:
            if "global" not in args.experiment_name:
                logging.info(
                    f"[seed {seed}] ckpt {ref_ckpt}: computing reference diagonal on full reference dataset '{ds_name}'..."
                )
                ref_dataset = data_train[ds_name]
                ref_loader = DataLoader(
                    ref_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                )
                reference_empirical_diag, _ = compute_empirical_fisher(
                    model, loss_func, ref_loader
                )

            dataset = data_train[ds_name]
            # Evaluates all samples per group without sampling or capping
            groups = build_demographic_groups_single_dataset(
                dataset, DEMOGRAPHICS, skip_missing=not args.keep_missing
            )
            logging.info(
                f"[seed {seed}] processing dataset '{ds_name}' ({len(groups)} subgroup(s))..."
            )

            for key, sub_ds in tqdm(
                list(groups.items()),
                desc=f"[seed {seed}] dataset '{ds_name}' subgroups",
                leave=False,
            ):
                if len(sub_ds) == 0:
                    continue

                loader = DataLoader(
                    sub_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                )

                sub_diag, n = compute_empirical_fisher(model, loss_func, loader)
                metrics = compute_gpu_diagonal_distances(
                    sub_diag, reference_empirical_diag
                )

                del sub_diag

                attr, code = key
                seed_results[(ds_name, attr, str(code))][ref_ckpt] = dict(
                    cosine_similarity=metrics["cosine_similarity"],
                    l1_distance=metrics["l1_distance"],
                    symmetrized_kl=metrics["symmetrized_kl"],
                    n=n,
                )

        del reference_empirical_diag
        free_model(model)
        gc.collect()
        torch.cuda.empty_cache()

        results[seed] = seed_results
        seed_path = write_seed_results(
            seed, seed_results, out_dir, args.experiment_name
        )
        logging.info(f"[seed {seed}] written: {seed_path}")

    merged_path = maybe_merge_seed_csvs(out_dir, args.experiment_name)
    if merged_path is not None and args.merge_into:
        merge_into_existing(merged_path, Path(args.merge_into))
    return results


# --------------------------------------------------------------------------- #
# CLI / entrypoint
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Curvature consistency sweep.")
    p.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        choices=[
            "single_dataset_over_time",
            "fisher_sim_last_all_datasets_global",
            "fisher_sim_last_all_datasets_local",
        ],
        help="Which experiment pipeline to run.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        help="Seed run ids, e.g. --seeds 123 2024 3407 (matches seed_run_{X} dirs)",
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=list(range(1, 12)),
        help="Checkpoint ids to process (default: 1..11)",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="single_dataset_over_time: which task's dataset to load "
        "(default: TASK_NAMES[0], i.e. 'ff++')",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="fisher_sim_last_all_datasets_*: only these datasets "
        "(default: all in TASK_NAMES), e.g. --datasets VQGAN Palette DCFACe",
    )
    p.add_argument(
        "--keep-missing",
        action="store_true",
        help="Also form groups for codes < 0 (missing annotation), as before",
    )
    p.add_argument(
        "--merge-into",
        type=str,
        default=None,
        help="Once all seeds are merged, replace these datasets in this "
        "existing merged CSV (backup <name>_before_merge.csv)",
    )
    p.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DATASET_DIR),
        help="Root dir containing one subfolder per task/generator (e.g. .../ai_face)",
    )
    p.add_argument(
        "--demo-csv",
        type=str,
        default=None,
        help="Path to demographics CSV with a path column "
        f"(default: {METADATA_CSV['train']})",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/consistency",
        help="Where to write per-seed and merged summary csv(s)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINT_DIR),
        help="Root dir containing seed_run_{seed}/ewc_{ckpt}.pt",
    )
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--log-file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_dir / "run.log"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    logging.info(
        f"experiment={args.experiment_name} seeds={args.seeds} checkpoints={args.checkpoints} "
        f"out_dir={out_dir}"
    )

    if args.experiment_name == "single_dataset_over_time":
        run_single_dataset_over_time_experiment(args)
    elif args.experiment_name in [
        "fisher_sim_last_all_datasets_global",
        "fisher_sim_last_all_datasets_local",
    ]:
        fisher_sim_last_all_datasets(args)
    else:
        raise ValueError(f"unknown experiment: {args.experiment_name}")

    logging.info("done.")


if __name__ == "__main__":
    main()
