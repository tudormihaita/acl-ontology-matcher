"""Command-line interface: ``python -m contrastive_om.cli <command> --config C``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import rdflib

from contrastive_om.config import load_config
from contrastive_om.ontology import (
    annotation_property_counts,
    load_from_config,
    ontology_stats,
    xref_counts,
)

log = logging.getLogger("contrastive_om")


def cmd_stats(args) -> None:
    cfg = config_from_args(args)
    out: dict = {}
    for which in ("source", "target"):
        path = Path(cfg.task[f"{which}_owl"])
        graph = rdflib.Graph()
        graph.parse(path, format="xml")
        ont = load_from_config(cfg, which, graph=graph)
        out[which] = {
            "file": str(path),
            "stats": ontology_stats(ont),
            "annotation_properties": annotation_property_counts(path, graph=graph),
            "cross_references": xref_counts(graph),
        }

        s = out[which]["stats"]
        print(f"\n== {which}: {path.name}")
        print(f"  classes {s['classes']} (alignable {s['alignable_classes']}, no label {s['classes_without_labels']})")
        print(
            f"  labels/class mean {s['labels_per_class_mean']} median {s['labels_per_class_median']} | "
            f">=2 labels {s['pct_ge2_labels']}% | exactly 1 label {s['pct_exactly1_label']}%"
        )
        print(f"  with definition {s['pct_with_definition']}% | with parent {s['pct_with_parent']}%")
        print(
            f"  depth (shortest) max {s['depth_max']} mean {s['depth_mean']} | "
            f"longest path max {s['longest_depth_max']} mean {s['longest_depth_mean']} | roots {s['roots']}"
        )
        print(
            f"  siblings/class mean {s['siblings_per_class_mean']} median {s['siblings_per_class_median']} "
            f"| with sibling {s['pct_with_sibling']}% | disjoint pairs {s['disjointness_pairs']}"
        )
        print("  predicates on classes (literal / iri / bnode):")
        for p, c in out[which]["annotation_properties"].items():
            print(f"    {c.get('literal', 0):7d} {c.get('iri', 0):7d} {c.get('bnode', 0):7d}  {p}")
        print(f"  cross-references: {out[which]['cross_references'] or 'none'}")

    out["config"] = {
        "ontology": cfg.ontology.to_dict(),
        "source_label_fixes": cfg.task.get("source_label_fixes"),
        "target_label_fixes": cfg.task.get("target_label_fixes"),
    }
    dest = Path(cfg.output.results_dir) / "data_stats.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


def parse_overrides(items: list[str] | None) -> dict:
    """``["pairs.positives=[syn, lex]", "seed=43"]`` -> nested dict (values parsed as YAML)."""
    import yaml

    out: dict = {}
    for item in items or []:
        key, _, value = item.partition("=")
        node = out
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return out


def config_from_args(args):
    return load_config(args.config, overrides=parse_overrides(getattr(args, "set", None)))


def load_task_ontologies(cfg) -> dict:
    onts = {which: load_from_config(cfg, which) for which in ("source", "target")}
    if cfg.task.get("subsample"):
        onts = subsample_task(cfg, onts)
    return onts


def subsample_task(cfg, onts: dict) -> dict:
    """Smoke configuration only: keep ``reference_pairs`` mapped pairs (so that matching has
    something to find) plus random classes, up to ``classes`` per ontology. Never used for
    reported results."""
    import random

    from contrastive_om.evaluate import read_mappings
    from contrastive_om.ontology import restrict

    sub = cfg.task.subsample
    rng = random.Random(int(sub.get("seed", 0)))
    refs = [(s, t) for s, t in read_mappings(cfg.task.full_refs)
            if s in onts["source"].classes and t in onts["target"].classes]
    chosen = rng.sample(sorted(refs), min(int(sub.reference_pairs), len(refs)))
    keep = {"source": {s for s, _ in chosen}, "target": {t for _, t in chosen}}
    for which, ont in onts.items():
        others = sorted(set(ont.classes) - keep[which])
        need = max(0, int(sub.classes) - len(keep[which]))
        keep[which] |= set(rng.sample(others, min(need, len(others))))
        onts[which] = restrict(ont, keep[which])
    log.info("smoke subsample: %d source / %d target classes", len(onts["source"]), len(onts["target"]))
    return onts


def mine_and_save(cfg, ontologies=None, force: bool = False):
    """Mine pairs (cached by config hash) and return (rows, stats, path)."""
    from contrastive_om.evaluate import read_mappings
    from contrastive_om.pairs import load_rows, mine_pairs, pairs_path, save_rows

    path = pairs_path(cfg)
    if path.exists() and not force:
        log.info("using cached pairs %s", path)
        return load_rows(path), json.loads(path.with_suffix(".stats.json").read_text()), path
    ontologies = ontologies or load_task_ontologies(cfg)
    refs = set(read_mappings(cfg.task.full_refs))  # only for the leakage assertion
    fixes = {w: cfg.task.get(f"{w}_label_fixes") for w in ontologies}
    rows, stats, lexicons = mine_pairs(cfg, ontologies, references=refs, label_fixes=fixes)
    save_rows(rows, stats, lexicons, path)
    return rows, stats, path


def cmd_mine(args) -> None:
    from contrastive_om.batching import batch_stats, batch_violations, build_batches

    cfg = config_from_args(args)
    rows, stats, path = mine_and_save(cfg, force=args.force)
    print(f"\npairs: {path} ({len(rows)} rows)")
    for name, s in stats.items():
        if not isinstance(s, dict):
            continue
        print(f"\n== {name}")
        for key in ("rows", "rows_by_source", "pct_classes_covered", "negatives_by_kind",
                    "rows_filled_with_random_negatives"):
            print(f"  {key}: {s[key]}")
        print(f"  single-label classes covered: {s['single_label_classes_covered']}/{s['single_label_classes']}")
        if "lexicon" in s:
            lx = s["lexicon"]
            print(f"  lexicon: {lx['rules_kept']} rules, {lx['variants_total']} variants "
                  f"{lx['variants_by_mechanism']}, single-label classes with variants "
                  f"{lx['single_label_classes_with_variants']}/{lx['single_label_classes']}")
    print(f"\nleakage check: {stats.get('leakage_check')}")

    bs = int(cfg.train.batch_size)
    batches = build_batches(rows, bs, int(cfg.seed), cfg.pairs.batch_scope)
    bad = [b for b in batches if batch_violations(rows, b, cfg.pairs.batch_scope)]
    print(f"batches (size {bs}): {batch_stats(batches, bs)} | batches violating rules: {len(bad)}")


def train_run(cfg, rows, run_dir: Path) -> dict:
    """Train, then verify that the saved checkpoint reloads to identical embeddings."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from contrastive_om.config import resolve_device
    from contrastive_om.train import loss_summary, read_train_log, train

    timing, model = train(cfg, rows, run_dir)
    summary = loss_summary(read_train_log(run_dir))

    device = resolve_device(cfg.get("device", "auto"))
    probe = [rows[0].anchor, rows[0].positive, "alzheimer disease"]
    trained = model.encode(probe, normalize_embeddings=True)
    reloaded = SentenceTransformer(str(run_dir / "model"), device=device).encode(probe, normalize_embeddings=True)
    summary.update({
        "checkpoint_reloads": bool(np.allclose(trained, reloaded, atol=1e-5)),
        "probe_positive_cosine": round(float(reloaded[0] @ reloaded[1]), 4),
    })
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return {**timing, **summary}


def cmd_train(args) -> None:
    from contrastive_om.train import make_run_dir

    cfg = config_from_args(args)
    rows, _, pairs_file = mine_and_save(cfg)
    run_dir = make_run_dir(cfg)
    out = train_run(cfg, rows, run_dir)
    print(f"\nrun: {run_dir}\npairs: {pairs_file}")
    for k, v in out.items():
        print(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="contrastive_om")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p):
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override a config value, e.g. --set pairs.positives='[syn, lex]'")
        return p

    p = with_config(sub.add_parser("stats", help="parse the ontologies and write data stats"))
    p.set_defaults(func=cmd_stats)

    p = with_config(sub.add_parser("mine", help="mine training pairs and print pair stats"))
    p.add_argument("--force", action="store_true", help="re-mine even if cached")
    p.set_defaults(func=cmd_mine)

    p = with_config(sub.add_parser("train", help="mine (cached) and train; saves to runs/<stamp>-<name>/"))
    p.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
