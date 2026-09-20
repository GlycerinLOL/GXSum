"""Smoke tests that run without GPU, network, or the heavy ML stack.

Heavy imports (torch, transformers, evaluate, datasets, tqdm) are stubbed so the
scripts can be imported and their pure-Python helpers exercised in CI.
"""
import importlib.util
import py_compile
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(ROOT.glob("*.py"))

HEAVY = ["torch", "torch.utils", "torch.utils.data", "transformers",
         "transformers.trainer_utils", "evaluate", "datasets", "tqdm"]


@pytest.fixture
def stubbed_heavy_imports(monkeypatch):
    for name in HEAVY:
        mod = types.ModuleType(name)
        mod.__getattr__ = lambda attr, _n=name: types.SimpleNamespace()  # any attribute resolves
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.syspath_prepend(str(ROOT))  # so `from load_gxsum import load_gxsum` resolves


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_compile(script):
    py_compile.compile(str(script), doraise=True)


def test_fine_tune_imports(stubbed_heavy_imports):
    _load(ROOT / "fine-tune.py", "fine_tune")


def test_compute_metrics_runs_on_tiny_batch(stubbed_heavy_imports):
    """Guards the arity bug fixed in 2496262: compute_metrics takes exactly three args."""
    ft = _load(ROOT / "fine-tune.py", "fine_tune")

    class Tok:
        pad_token_id = 0
        def batch_decode(self, ids, **_):
            return [" ".join(f"w{t}" for t in row if t != self.pad_token_id) for row in ids]

    class Rouge:
        def compute(self, predictions, references, **_):
            assert len(predictions) == len(references) == 2
            return {"rouge1": 0.5, "rouge2": 0.25, "rougeL": 0.5}

    preds = np.array([[5, 6, 7, 0], [8, 9, -100, -100]])
    labels = np.array([[5, 6, 0, 0], [8, -100, -100, -100]])

    out = ft.compute_metrics((preds, labels), Tok(), Rouge())

    assert out["rouge1"] == 0.5
    assert out["gen_len"] == pytest.approx(2.5)  # [3 non-pad, 2 non-pad] after -100 -> pad


def test_summarization_name_mapping_covers_paper_datasets(stubbed_heavy_imports):
    ft = _load(ROOT / "fine-tune.py", "fine_tune")
    for ds in ("xsum", "cnn_dailymail", "reddit_tifu"):
        src, tgt = ft.summarization_name_mapping[ds]
        assert src and tgt
