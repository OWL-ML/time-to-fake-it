"""Routed inference for prompt-based CL methods (SOYO, PINA, S-Prompts).

The DER++ ``run_inference`` calls ``model(images)`` directly. Prompt
methods cannot: at test time the task is unknown, so the model must
first pick a prompt —

    feat      = model.extract_vector(images)
    selection = trainer._knn_select(feat)        # per-sample routing
    logits    = model.interface(images, selection)

— which is exactly the path ``PromptCLTrainer.evaluate`` uses. This
module mirrors that, and emits the SAME per-sample record schema as the
DER++ version so both pipelines feed one analysis stack, plus four
routing columns that come free from the same pass:

    routed_task, true_task, routing_confidence, routing_correct

That means routing accuracy per subgroup and prediction accuracy per
subgroup are measured on the same data, in the same pass, and join
row-for-row.

LABEL CONVENTION — read before using
------------------------------------
The record schema is canonical: ``label`` 1 = fake, ``prediction``
1 = fake, ``probability`` = P(fake). The two pipelines do NOT agree on
the raw index that means "fake":

* DER++: dataset gives 1 = fake, model class 0 = fake, so its
  run_inference flips the prediction only.
* Prompt trainers: ``evaluate`` compares ``logits.argmax(1)`` to the
  target directly, so model class k == dataset class k. With an
  ImageFolder-style dataset the folder order is alphabetical, which
  usually puts fake at index 0 — meaning BOTH label and prediction need
  flipping to reach the canonical convention.

``fake_index`` states which index means fake in this pipeline; both the
label and the prediction are mapped from it. VERIFY it against your
class order (``dataset.class_to_idx``) before trusting the TPR/FPR
numbers — getting it wrong silently swaps TPR and FPR rather than
raising anything. Set ``fake_index=None`` to write the raw indices
through unmapped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from .demographics import ATTRIBUTES, demo_value
from .subgroup_metrics import collect_boundary

TEMPERATURE = 5.0

# Demographic columns are the canonical names from demographics.py, read
# from the dataset's dict via demo_value so the exact key spelling
# ("Predicted Gender" vs "gender") does not matter. All five attributes
# are emitted, including BOTH skin-tone columns: the 10-level scale and
# the 3-bin group are separate attributes, and intersection is built from
# the 3-bin one.
SAMPLE_COLS = [
    "image_path",
    "label",
    "prediction",
    "probability",
    "probability_t5",
    "logit_fake",
    "logit_real",
    *ATTRIBUTES,
    "routed_task",
    "true_task",
    "routing_confidence",
    "routing_correct",
]


def _demographics_of(dataset, idx: int) -> Optional[Dict]:
    """Fetch one sample's demographics, whichever accessor exists."""
    for name in ("get_demographics", "get_demographic_attributes"):
        fn = getattr(dataset, name, None)
        if fn is not None:
            return fn(int(idx))
    return None


def _path_of(dataset, idx: int) -> str:
    """Image path for a sample; falls back to the index."""
    for attr in ("samples", "imgs"):
        seq = getattr(dataset, attr, None)
        if seq is not None:
            try:
                return str(seq[int(idx)][0])
            except (IndexError, TypeError):
                break
    fn = getattr(dataset, "get_path", None)
    return str(fn(int(idx))) if fn else str(int(idx))


def routing_confidence(trainer, feat) -> np.ndarray:
    """Per-sample confidence of the routing decision.

    SOYO exposes a trained softmax over domains (clamped to seen tasks);
    PINA / S-Prompts expose _routing_scores (negative L1 distance), whose
    softmax is an uncalibrated ranking strength — comparable within a
    method, not across methods. NaN when neither is available.
    """
    scorer = getattr(trainer, "_routing_scores", None)
    if scorer is not None:
        return F.softmax(scorer(feat), dim=1).max(dim=1)[0].cpu().numpy()
    if hasattr(trainer, "soyo_network"):
        dl = trainer.soyo_network(feat.to(torch.float32))
        dl = dl[:, : trainer.model.numtask]
        return F.softmax(dl, dim=1).max(dim=1)[0].cpu().numpy()
    return np.full(feat.shape[0], np.nan, dtype=float)


def records_from_batch(
    dataset,
    idxs,
    labels,
    logits,
    selection,
    route_conf,
    true_task: int,
    fake_index: Optional[int] = 0,
    dataset_dir: Optional[str] = None,
) -> List[Dict]:
    """Build per-sample record rows for one batch.

    Shared by run_inference_prompt (standalone / post-hoc use) and
    PromptCLTrainer.evaluate (folded into the normal eval pass), so the
    record schema and the label convention live in exactly one place.

    ``fake_index`` states which class index means fake in this pipeline;
    label and prediction are mapped to the canonical 1 = fake. Pass None
    to write raw indices through.
    """
    probs = F.softmax(logits, dim=1)
    probs_t = F.softmax(logits / TEMPERATURE, dim=1)
    argmax = logits.argmax(dim=1)

    logits_np = logits.float().cpu().numpy()
    probs_np = probs.float().cpu().numpy()
    probs_t_np = probs_t.float().cpu().numpy()
    argmax_np = argmax.cpu().numpy()
    labels_np = labels.cpu().numpy()
    routed_np = selection.cpu().numpy()

    if fake_index is None:
        pred_np, lab_np, fake_col, real_col = argmax_np, labels_np, 0, 1
    else:
        fake_col = int(fake_index)
        real_col = 1 - fake_col
        pred_np = (argmax_np == fake_col).astype(int)
        lab_np = (labels_np == fake_col).astype(int)

    rows = []
    for b in range(len(labels_np)):
        idx = int(idxs[b])
        dem = _demographics_of(dataset, idx) or {}
        path = _path_of(dataset, idx)
        if dataset_dir:
            path = path.replace(f"{dataset_dir}/", "")
        rows.append(
            {
                "image_path": path,
                "label": int(lab_np[b]),
                "prediction": int(pred_np[b]),
                "probability": float(probs_np[b, fake_col]),  # P(fake)
                "probability_t5": float(probs_t_np[b, fake_col]),
                "logit_fake": float(logits_np[b, fake_col]),
                "logit_real": float(logits_np[b, real_col]),
                **{attr: demo_value(dem, attr) for attr in ATTRIBUTES},
                "routed_task": int(routed_np[b]),
                "true_task": int(true_task),
                "routing_confidence": float(route_conf[b]),
                "routing_correct": int(routed_np[b] == true_task),
            }
        )
    return rows


def finalize_records(rows: List[Dict], embeddings=None, true_task=None):
    """Assemble rows into a DataFrame, dropping samples without
    demographics while keeping any embeddings aligned."""
    records = pd.DataFrame(rows, columns=SAMPLE_COLS)
    valid = records["intersection"].notna().to_numpy()
    if len(records) and not valid.all():
        logger.warning(
            "[eval] task {}: dropping {} rows without demographics",
            true_task,
            int((~valid).sum()),
        )
        records = records[valid].reset_index(drop=True)
        if embeddings is not None:
            embeddings = embeddings[valid]
    return records, embeddings


@torch.no_grad()
def run_inference_prompt(
    trainer,
    dataset,
    true_task: int,
    batch_size: int = 128,
    num_workers: int = 4,
    extract_embeddings: bool = False,
    fake_index: Optional[int] = 0,
    dataset_dir: Optional[str] = None,
):
    """Routed inference over one task's dataset (standalone pass).

    Use this for post-hoc evaluation from saved checkpoints. During
    training the same records are produced for free by
    PromptCLTrainer.evaluate, which folds record collection into the
    eval pass it already runs.

    Args:
        trainer: a PromptCLTrainer subclass with routing state already
            built (i.e. after _after_task, or after load_checkpoint).
        dataset: IndxImageFolder yielding (idx, image, label).
        true_task: the task this dataset belongs to — ground truth for
            routing correctness.
        fake_index: index that means "fake" in this pipeline's class
            order; see records_from_batch.

    Returns:
        (records_df, embeddings)
    """
    model, device = trainer.model, trainer.device
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(getattr(device, "type", str(device)) == "cuda"),
    )

    rows, emb_chunks = [], []
    for idxs, images, labels in loader:
        images = images.to(device, non_blocking=True)

        # routing: identical to PromptCLTrainer.evaluate
        feat = model.extract_vector(images)
        selection = trainer._knn_select(feat)
        out = model.interface(images, selection)
        logits = out["logits"] if isinstance(out, dict) else out

        if extract_embeddings:
            emb_chunks.append(feat.detach().half().cpu().numpy())

        rows += records_from_batch(
            dataset,
            idxs,
            labels,
            logits,
            selection,
            routing_confidence(trainer, feat),
            true_task,
            fake_index=fake_index,
            dataset_dir=dataset_dir,
        )

    embeddings = np.concatenate(emb_chunks) if emb_chunks else None
    return finalize_records(rows, embeddings, true_task)


def collect_boundary_prompt(
    trainer,
    boundary: int,
    out_dir: str | Path,
    datasets: Optional[Sequence] = None,
    **kwargs,
) -> List[Path]:
    """Run routed inference on every seen task and store the records.

    Writes the same parquet layout as subgroup_metrics.collect_boundary,
    so build_tables() consumes both pipelines identically. Call after
    _after_task (routing state must exist for all seen tasks).
    """
    datasets = datasets if datasets is not None else trainer.test_datasets
    written: List[Path] = []
    for t in range(boundary + 1):
        written += collect_boundary(
            boundary=boundary,
            datasets=[datasets[t]],
            data_tasks=[t],
            predict=lambda ds, t=t: run_inference_prompt(
                trainer, ds, true_task=t, **kwargs
            ),
            out_dir=out_dir,
        )
    return written
