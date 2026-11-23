"""Contrastive (InfoNCE) training of a sentence-transformers bi-encoder (SPEC §4.3).

A small explicit training loop is used instead of the high-level Trainer, so the pre-built
batches of ``batching.py`` (class-unique, per-ontology, ...) are consumed exactly as built.

Per step: tokenize the columns (anchors, positives, negative_1..k), compute the
InfoNCE loss (scale 1/τ over cosine similarities), AdamW
update with linear warmup/decay. fp32 throughout (no mixed precision on MPS).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import yaml

from contrastive_om.batching import batch_violations, build_batches
from contrastive_om.config import resolve_device, set_seed
from contrastive_om.pairs import PairRow

log = logging.getLogger(__name__)

try:  # sentence-transformers >= 5 moved the losses
    from sentence_transformers.sentence_transformer import losses as st_losses
except ImportError:  # pragma: no cover
    from sentence_transformers import losses as st_losses


def make_run_dir(cfg, root: str | Path | None = None) -> Path:
    """``runs/<YYYYmmdd-HHMM>-<name>/`` with a copy of the resolved config."""
    root = Path(root or cfg.output.runs_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = root / f"{stamp}-{cfg.name}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}-{cfg.name}-{suffix}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return run_dir


def load_encoder(cfg, device: str):
    from sentence_transformers import SentenceTransformer

    # On MPS, the fused SDPA attention kernel has no dropout support, which breaks the cached
    # loss; eager attention computes the same thing. Used for every loss so runs stay comparable.
    model_kwargs = {"attn_implementation": "eager"} if device == "mps" else None
    model = SentenceTransformer(cfg.model.encoder, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = int(cfg.model.max_seq_length)
    return model


def build_loss(model, cfg_train):
    """InfoNCE losses, implemented by sentence-transformers' "Multiple Negatives Ranking" losses:

    - ``infonce``: softmax over cos(anchor, candidate) / τ, candidates = all positives and hard
      negatives of the batch; cross-entropy with the true positive as target;
    - ``infonce_symmetric``: also the reverse direction (positive -> anchor), as in CLIP;
    - ``infonce_cached``: same loss as ``infonce``, computed in mini-batches with gradient caching
      so very large batches fit in memory.
    """
    scale = 1.0 / float(cfg_train.temperature)
    name = cfg_train.loss
    if name == "infonce":
        return st_losses.MultipleNegativesRankingLoss(model, scale=scale)
    if name == "infonce_symmetric":
        return st_losses.MultipleNegativesSymmetricRankingLoss(model, scale=scale)
    if name == "infonce_cached":
        mini = int(cfg_train.get("mini_batch_size", 32))
        return st_losses.CachedMultipleNegativesRankingLoss(model, scale=scale, mini_batch_size=mini)
    raise ValueError(f"unsupported loss '{name}' (supcon is a 'should' item, not implemented yet)")


def _features(model, texts: list[str], device: str) -> dict[str, torch.Tensor]:
    # sentence-transformers >= 6 renamed tokenize() to preprocess()
    feats = model.preprocess(texts) if hasattr(model, "preprocess") else model.tokenize(texts)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}


def _optimizer(model, cfg_train, total_steps: int):
    from transformers import get_linear_schedule_with_warmup

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm")
    params = list(model.named_parameters())
    groups = [
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)],
         "weight_decay": float(cfg_train.get("weight_decay", 0.01))},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=float(cfg_train.lr))
    warmup = int(round(float(cfg_train.warmup_ratio) * total_steps))
    return opt, get_linear_schedule_with_warmup(opt, warmup, total_steps)


def train(cfg, rows: Sequence[PairRow], run_dir: Path):
    """Train the configured encoder on ``rows``, save it to ``run_dir/model``.

    Returns ``(timing, model)``.
    """
    seed = int(cfg.seed)
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    scope = cfg.pairs.batch_scope
    batch_size = int(cfg.train.batch_size)
    epochs = int(cfg.train.epochs)

    epoch_batches = []
    for epoch in range(epochs):  # a fresh, rule-abiding batch order per epoch
        batches = build_batches(rows, batch_size, seed + epoch, scope)
        bad = sum(1 for b in batches if batch_violations(rows, b, scope))
        if bad:
            raise RuntimeError(f"{bad} batches violate the batch rules")
        epoch_batches.append(batches)
    total_steps = sum(len(b) for b in epoch_batches)

    model = load_encoder(cfg, device)
    loss_fn = build_loss(model, cfg.train)
    opt, sched = _optimizer(model, cfg.train, total_steps)
    max_grad_norm = float(cfg.train.get("max_grad_norm", 1.0))

    log.info("training %s on %s: %d rows, %d steps, batch %d, loss %s, tau %s",
             cfg.model.encoder, device, len(rows), total_steps, batch_size, cfg.train.loss, cfg.train.temperature)
    model.train()
    step, t0 = 0, time.perf_counter()
    with open(run_dir / "train_log.jsonl", "w") as log_fh:
        for epoch, batches in enumerate(epoch_batches):
            for batch in batches:
                chosen = [rows[i] for i in batch]
                k = len(chosen[0].negatives)
                columns = [[r.anchor for r in chosen], [r.positive for r in chosen]]
                columns += [[r.negatives[j] for r in chosen] for j in range(k)]
                features = [_features(model, col, device) for col in columns]
                labels = torch.zeros(len(chosen), device=device)  # unused by the InfoNCE losses

                loss = loss_fn(features, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

                step += 1
                record = {"step": step, "epoch": epoch, "loss": float(loss.detach()), "lr": sched.get_last_lr()[0],
                          "batch_size": len(chosen), "elapsed_s": round(time.perf_counter() - t0, 2)}
                log_fh.write(json.dumps(record) + "\n")
                if step % 50 == 0 or step == total_steps:
                    log.info("step %d/%d loss %.4f", step, total_steps, record["loss"])
    train_time = time.perf_counter() - t0

    model.eval()
    model_dir = run_dir / "model"
    model.save(str(model_dir))
    timing = {"train_s": round(train_time, 2), "steps": total_steps, "rows": len(rows), "device": device}
    if device == "mps":
        timing["mps_driver_memory_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 1)
    elif device == "cuda":
        timing["cuda_peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return timing, model


def read_train_log(run_dir: Path) -> list[dict]:
    with open(Path(run_dir) / "train_log.jsonl") as fh:
        return [json.loads(line) for line in fh]


def loss_summary(records: list[dict]) -> dict:
    """Mean loss over the first and last 10% of steps (loss-decrease check)."""
    losses = [r["loss"] for r in records]
    n = max(1, len(losses) // 10)
    first, last = sum(losses[:n]) / n, sum(losses[-n:]) / n
    return {"loss_first_10pct": round(first, 4), "loss_last_10pct": round(last, 4), "loss_decreased": last < first}
