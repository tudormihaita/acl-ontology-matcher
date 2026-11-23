"""Command-line interface: ``python -m contrastive_om.cli <command> ...``.

Commands: stats, mine, train, evaluate, run, match (demo on any two OWL files), inspect.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import rdflib

from contrastive_om.config import load_config
from contrastive_om.ontology import annotation_property_counts, load_from_config, ontology_stats, xref_counts

log = logging.getLogger("contrastive_om")


# --------------------------------------------------------------------------- config


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
    """``--run DIR`` uses the run's saved config; ``--config`` a YAML file. ``--set`` overrides both."""
    run_dir = getattr(args, "run", None)
    path = Path(run_dir) / "config.yaml" if run_dir else args.config
    if not path:
        raise SystemExit("give --config or --run")
    return load_config(path, overrides=parse_overrides(getattr(args, "set", None)))


def _print_metrics(m: dict) -> None:
    def g(k):
        v = m.get(k)
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print(f"  θ = {g('theta')}  ({m.get('n_mappings')} mappings)")
    print(f"  test  P/R/F1 {g('test_precision')} / {g('test_recall')} / {g('test_f1')}"
          f"   (95% CI F1 {g('test_f1_ci_low')}-{g('test_f1_ci_high')})")
    print(f"  val   P/R/F1 {g('val_precision')} / {g('val_recall')} / {g('val_f1')}")
    print(f"  local ranking  MRR {g('local_mrr')}  Hits@1 {g('local_hits@1')}  Hits@5 {g('local_hits@5')}")
    print(f"  full retrieval MRR {g('full_mrr')}  R@1 {g('full_hits@1')}  R@5 {g('full_hits@5')}  R@10 {g('full_hits@10')}")
    if "alignment" in m:
        print(f"  alignment {g('alignment')}  uniformity {g('uniformity')}")
    print(f"  times: " + ", ".join(f"{k} {m[k]}" for k in ("train_s", "encode_s", "match_s") if k in m))


# --------------------------------------------------------------------------- commands


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
        print(f"  labels/class mean {s['labels_per_class_mean']} median {s['labels_per_class_median']} | "
              f">=2 labels {s['pct_ge2_labels']}% | exactly 1 label {s['pct_exactly1_label']}%")
        print(f"  with definition {s['pct_with_definition']}% | with parent {s['pct_with_parent']}%")
        print(f"  depth (shortest) max {s['depth_max']} mean {s['depth_mean']} | "
              f"longest path max {s['longest_depth_max']} mean {s['longest_depth_mean']} | roots {s['roots']}")
        print(f"  siblings/class mean {s['siblings_per_class_mean']} median {s['siblings_per_class_median']} "
              f"| with sibling {s['pct_with_sibling']}% | disjoint pairs {s['disjointness_pairs']}")
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


def cmd_mine(args) -> None:
    from contrastive_om.batching import batch_stats, batch_violations, build_batches
    from contrastive_om.pipeline import mine_and_save

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


def cmd_train(args) -> None:
    from contrastive_om.pipeline import mine_and_save, train_run
    from contrastive_om.train import make_run_dir

    cfg = config_from_args(args)
    rows, _, pairs_file = mine_and_save(cfg)
    run_dir = make_run_dir(cfg)
    summary, _ = train_run(cfg, rows, run_dir)
    print(f"\nrun: {run_dir}\npairs: {pairs_file}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_evaluate(args) -> None:
    from contrastive_om.pipeline import evaluate_run

    cfg = config_from_args(args)
    metrics = evaluate_run(cfg, Path(args.run))
    print(f"\nevaluated {args.run} ({cfg.name})")
    _print_metrics(metrics)


def cmd_run(args) -> None:
    from contrastive_om.pipeline import run

    cfg = config_from_args(args)
    run_dir, metrics = run(cfg)
    print(f"\nrun: {run_dir} ({cfg.name})")
    _print_metrics(metrics)


def cmd_match(args) -> None:
    """Demo: match any two OWL files with a (trained) model and write mappings.tsv."""
    from contrastive_om.evaluate import apply_threshold, write_scored_mappings
    from contrastive_om.match import match
    from contrastive_om.pipeline import _alignable_mask, _parent_indices
    from contrastive_om.represent import embed_ontology
    from contrastive_om.train import load_encoder

    cfg = config_from_args(args)
    src = load_from_config(cfg, "source", path=args.source) if args.source else load_from_config(cfg, "source")
    tgt = load_from_config(cfg, "target", path=args.target) if args.target else load_from_config(cfg, "target")
    model = load_encoder(cfg, path=args.model)
    src_emb, tgt_emb = embed_ontology(src, model), embed_ontology(tgt, model)
    result = match(src_emb, tgt_emb, _alignable_mask(src, src_emb), _alignable_mask(tgt, tgt_emb), cfg.match,
                   _parent_indices(src, src_emb), _parent_indices(tgt, tgt_emb))
    mappings = apply_threshold(result.mappings, args.threshold) if args.threshold is not None else result.mappings
    write_scored_mappings(mappings, args.out)
    print(f"{len(mappings)} mappings ({src.name} -> {tgt.name}) written to {args.out}")
    for (s, t), v in sorted(mappings.items(), key=lambda kv: -kv[1])[: args.show]:
        print(f"  {v:.3f}  {src[s].labels[0]!r:45} -> {tgt[t].labels[0]!r}")


def cmd_inspect(args) -> None:
    """Demo aid: nearest target classes for a free-text label."""
    from contrastive_om.ontology import normalize_label
    from contrastive_om.represent import embed_ontology, encode_texts
    from contrastive_om.train import load_encoder

    cfg = config_from_args(args)
    tgt = load_from_config(cfg, "target", path=args.target) if args.target else load_from_config(cfg, "target")
    model = load_encoder(cfg, path=args.model)
    emb = embed_ontology(tgt, model)
    q = encode_texts(model, [normalize_label(args.query)])[0]
    import numpy as np

    label_scores = emb.label_vecs @ q
    class_scores = np.maximum.reduceat(label_scores, emb.offsets)
    for i in np.argsort(-class_scores)[: args.k]:
        best = emb.labels[emb.label_slice(i)][int(np.argmax(label_scores[emb.label_slice(i)]))]
        print(f"  {class_scores[i]:.3f}  {tgt[emb.iris[i]].labels[0]!r:50} (via {best!r})  {emb.iris[i]}")


def cmd_summary(args) -> None:
    """Compare runs from results/summary.csv. The winner is chosen on VALIDATION F1 (ties:
    validation full-retrieval MRR); test columns are shown for information only."""
    import pandas as pd

    df = pd.read_csv(args.file)
    if args.names:
        df = df[df["name"].isin(args.names)]
    if args.where:
        for cond in args.where:
            key, _, value = cond.partition("=")
            df = df[df[key].astype(str) == value]
    if df.empty:
        print("no matching rows")
        return
    df = df.sort_values(["val_f1", "val_full_mrr"], ascending=False)
    cols = ["name", "selection", "scoring", "class_repr", "neighbor_lambda", "theta",
            "val_f1", "val_full_mrr", "test_precision", "test_recall", "test_f1", "local_mrr", "full_hits@1"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(df[cols].to_string(index=False))
    print(f"\nchosen on validation: {df.iloc[0]['name']} "
          f"(val F1 {df.iloc[0]['val_f1']:.3f}, val full MRR {df.iloc[0]['val_full_mrr']:.3f})")


def cmd_analyze(args) -> None:
    from contrastive_om.analysis import run_analysis

    out = run_analysis(args.results)
    for name, a in out["categories"].items():
        print(f"\n== {name}: θ {a['theta']:.3f}, {a['n_predictions']} predictions, "
              f"{a['false_positives']} FP, {a['false_negatives']} FN")
        print(f"  FP categories: {a['fp_categories']}  (gold label overlap >= 0.5: {a['fp_with_gold_high_label_overlap']})")
        print(f"  FN gold rank: {a['fn_gold_rank']} | predicted another target {a['fn_predicted_another_target']} "
              f"| single-label src {a['fn_single_label_source']} tgt {a['fn_single_label_target']}")
    print(f"\nUMAP subtree: {out['umap_subtree']} {out['umap']}")
    print("figures:", *out["figures"], sep="\n  ")


# --------------------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="contrastive_om")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p, required=True):
        p.add_argument("--config", required=required)
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

    p = with_config(sub.add_parser("evaluate", help="evaluate a run dir (uses its config; --set for E4/E5)"),
                    required=False)
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = with_config(sub.add_parser("run", help="mine -> train -> match -> evaluate"))
    p.set_defaults(func=cmd_run)

    p = with_config(sub.add_parser("match", help="demo: match two OWL files"))
    p.add_argument("--model", required=True, help="trained model dir, e.g. runs/<run>/model")
    p.add_argument("--source"), p.add_argument("--target")
    p.add_argument("--threshold", type=float)
    p.add_argument("--out", default="mappings.tsv")
    p.add_argument("--show", type=int, default=10)
    p.set_defaults(func=cmd_match)

    p = with_config(sub.add_parser("inspect", help="demo: top-k target classes for a label"))
    p.add_argument("--model", required=True)
    p.add_argument("--target")
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("summary", help="compare runs in results/summary.csv (winner on validation F1)")
    p.add_argument("--names", nargs="*", help="config names to compare")
    p.add_argument("--where", nargs="*", help="column=value filters, e.g. selection=argmax")
    p.add_argument("--file", default="results/summary.csv")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("analyze", help="Phase 6: error categories, examples, lexicon examples, figures")
    p.add_argument("--results", default="results")
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
