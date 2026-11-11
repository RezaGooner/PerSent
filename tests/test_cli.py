from __future__ import annotations

import io
import json
import os

import pandas as pd
import pytest

from PerSent import __version__
from PerSent.cli import main

DATASET = os.path.join(os.path.dirname(__file__), os.pardir, "dataset", "emotion_dataset.csv")


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and __version__ in capsys.readouterr().out


def test_polarity_text(capsys):
    code, out, _ = run(capsys, "polarity", "کیفیت عالی داشت", "افتضاح بود")
    assert code == 0
    assert [line.split("\t")[0] for line in out.splitlines()] == ["recommended", "not_recommended"]


def test_polarity_json_and_proba(capsys):
    _, out, _ = run(capsys, "polarity", "کیفیت عالی داشت", "--json")
    data = json.loads(out)
    assert data["label"] == "recommended" and abs(sum(data["scores"].values()) - 1) < 1e-6
    _, out, _ = run(capsys, "polarity", "کیفیت عالی داشت", "--proba")
    assert "recommended=" in out


def test_polarity_stdin_and_file(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO("کیفیت عالی داشت\n\nافتضاح بود\n"))
    _, out, _ = run(capsys, "polarity")
    assert len(out.splitlines()) == 2
    f = tmp_path / "t.txt"
    f.write_text("کیفیت عالی داشت\n", encoding="utf-8")
    _, out, _ = run(capsys, "polarity", "--file", str(f))
    assert out.startswith("recommended")


def test_emotion(capsys):
    _, out, _ = run(capsys, "emotion", "خیلی خوشحالم", "--lang", "en")
    assert out.startswith("happiness")
    _, out, _ = run(capsys, "emotion", "سلام", "--json")
    assert json.loads(out)["dominant"] is None


def test_csv_commands(capsys, tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"نظرات": ["کیفیت عالی داشت", "افتضاح بود"]}).to_csv(src, index=False)
    out, summary = tmp_path / "o.csv", tmp_path / "s.csv"
    code, _, err = run(capsys, "csv", str(src), "-o", str(out), "-c", "نظرات", "--scores", "--summary", str(summary))
    assert code == 0 and "2 rows" in err
    assert "score_recommended" in pd.read_csv(out, encoding="utf-8-sig").columns and summary.exists()
    code, _, _ = run(capsys, "csv", str(src), "-o", str(out), "-t", "emotion", "-c", "0", "-q", "--lang", "en")
    assert code == 0 and "dominant_emotion" in pd.read_csv(out, encoding="utf-8-sig").columns


def test_errors_return_1(capsys, tmp_path):
    code, _, err = run(capsys, "csv", str(tmp_path / "missing.csv"), "-o", str(tmp_path / "o.csv"))
    assert code == 1 and err.startswith("error:")
    code, _, err = run(capsys, "polarity", "x", "-m", str(tmp_path))
    assert code == 1


def test_train_and_evaluate(capsys, comments_csv, tmp_path):
    model = tmp_path / "m"
    code, _, err = run(capsys, "train", "polarity", str(comments_csv), "-o", str(model), "-q")
    assert code == 0 and "accuracy" in err and (model / "polarity_model.npz").exists()
    w2v = tmp_path / "w2v"
    code, _, _ = run(
        capsys, "train", "polarity", str(comments_csv), "-o", str(w2v), "--method", "word2vec", "--vector-size", "8"
    )
    assert code == 0 and (w2v / "comment_model.npz").exists()
    code, out, _ = run(capsys, "evaluate", str(comments_csv), "-m", str(model), "--json")
    assert code == 0 and json.loads(out)["n"] == 30
    _, out, _ = run(capsys, "evaluate", str(comments_csv), "-m", str(model))
    assert "macro-F1" in out and "precision" in out
    _, out, _ = run(capsys, "polarity", "خیلی خوب بود", "-m", str(model))
    assert out.split("\t")[0] in {"recommended", "not_recommended", "no_idea"}


def test_train_emotion(capsys, emotion_csv, tmp_path):
    code, _, err = run(capsys, "train", "emotion", str(emotion_csv), "-o", str(tmp_path / "e.json"), "-q")
    assert code == 0 and "3 rows" in err
    _, out, _ = run(capsys, "emotion", "از تاریکی می‌ترسم", "-m", str(tmp_path / "e.json"))
    assert out.startswith("ترس")


def test_named_models_and_emotion_classifier(capsys, tmp_path):
    _, out, _ = run(capsys, "polarity", "کیفیت عالی داشت", "-m", "word2vec-v1")
    assert out.startswith("recommended")
    data = pd.DataFrame(
        {"text": ["خیلی خوشحال و شادم", "از تاریکی می‌ترسم", "خیلی غمگینم", "ساعت پنج میام"] * 6,
         "sentiment": ["شادی", "ترس", "غم", "other"] * 6}
    )  # fmt: skip
    csv = tmp_path / "emo.csv"
    data.to_csv(csv, index=False)
    code, _, err = run(capsys, "train", "emotion", str(csv), "-o", str(tmp_path / "e.npz"), "--method", "classifier")
    assert code == 0 and "accuracy" in err
    code, out, _ = run(capsys, "evaluate", str(csv), "--task", "emotion", "-m", str(tmp_path / "e.npz"))
    assert code == 0 and "macro-F1" in out
    _, out, _ = run(capsys, "emotion", "خیلی خوشحال و شادم", "-m", str(tmp_path / "e.npz"))
    assert out.startswith("شادی")
