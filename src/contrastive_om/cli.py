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
    cfg = load_config(args.config)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="contrastive_om")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="parse the ontologies and write data stats")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
