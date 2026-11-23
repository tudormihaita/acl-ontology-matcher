"""Training helpers that do not need a model download (the full loop is checked by the smoke run)."""

import json

import pytest

from contrastive_om.cli import parse_overrides
from contrastive_om.config import Config, load_config
from contrastive_om.ontology import Ontology, OntologyClass, restrict
from contrastive_om.train import loss_summary, make_run_dir, read_train_log


def test_loss_summary():
    records = [{"loss": x} for x in [4.0] * 10 + [2.0] * 80 + [1.0] * 10]
    s = loss_summary(records)
    assert s == {"loss_first_10pct": 4.0, "loss_last_10pct": 1.0, "loss_decreased": True}


def test_make_run_dir_is_unique_and_stores_config(tmp_path):
    cfg = Config({"name": "x", "seed": 1, "output": {"runs_dir": str(tmp_path)}})
    a, b = make_run_dir(cfg), make_run_dir(cfg)
    assert a != b and a.name.endswith("-x")
    assert "seed: 1" in (a / "config.yaml").read_text()


def test_read_train_log(tmp_path):
    (tmp_path / "train_log.jsonl").write_text(json.dumps({"step": 1, "loss": 2.5}) + "\n")
    assert read_train_log(tmp_path) == [{"step": 1, "loss": 2.5}]


def test_parse_overrides():
    out = parse_overrides(["pairs.positives=[syn, lex]", "seed=43", "train.temperature=0.1", "name=e1"])
    assert out == {"pairs": {"positives": ["syn", "lex"]}, "seed": 43, "train": {"temperature": 0.1}, "name": "e1"}


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/smoke.yaml", overrides={"train": {"loss": "infonce_cached"}})
    assert cfg.name == "smoke"
    assert cfg.train.batch_size == 32  # from smoke.yaml
    assert cfg.train.lr == 2.0e-5  # inherited from base.yaml
    assert cfg.train.loss == "infonce_cached"  # override


def test_restrict_drops_edges_to_removed_classes():
    cls = {k: OntologyClass(iri=k, labels=[k]) for k in ("p", "a", "b")}
    for child in ("a", "b"):
        cls[child].parents.add("p")
        cls["p"].children.add(child)
    small = restrict(Ontology("t", cls), {"a", "b"})
    assert set(small.classes) == {"a", "b"}
    assert small["a"].parents == set() and small.siblings("a") == set()


def test_unknown_loss_is_rejected():
    from contrastive_om.train import build_loss

    with pytest.raises(ValueError):
        build_loss(model=None, cfg_train=Config({"loss": "supcon", "temperature": 0.05}))
