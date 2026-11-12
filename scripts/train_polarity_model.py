"""Train the bundled ``tfidf-v2`` polarity model and print its evaluation.

    python scripts/build_polarity_dataset.py big_train.csv data/polarity
    python scripts/train_polarity_model.py data/polarity src/PerSent/models/polarity_tfidf_v2.npz

The dataset directory must contain train.csv / valid.csv / test.csv (columns
``body`` and ``recommendation_status``) as produced by build_polarity_dataset.py.
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PerSent import PolarityClassifier, __version__

# chosen on the validation split (see docs/MODEL_CARD.md)
CONFIG = {
    "word_ngrams": (1, 2),
    "char_ngrams": (2, 5),
    "max_word_features": 200_000,
    "max_char_features": 100_000,
    "min_df": 3,
}
C = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()

    train = pd.read_csv(a.data / "train.csv")
    valid = pd.read_csv(a.data / "valid.csv")
    test = pd.read_csv(a.data / "test.csv")

    t = time.time()
    clf = PolarityClassifier()
    clf.fit(train.body, train.recommendation_status, test_size=0, C=C, **CONFIG)
    print(f"trained on {len(train)} texts in {time.time() - t:.0f}s")

    metrics = {}
    for name, df in (("valid", valid), ("test", test)):
        rep = clf.evaluate(df.body, df.recommendation_status)
        metrics[name] = {"n": rep.n, "accuracy": round(rep.accuracy, 4), "macro_f1": round(rep.macro_f1, 4)}
        print(name, rep)
        for label, m in rep.per_label.items():
            print(f"   {label:<16} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    test_rep = clf.evaluate(test.body, test.recommendation_status)
    metrics["test"]["per_label"] = {k: {m: round(v, 4) for m, v in d.items()} for k, d in test_rep.per_label.items()}

    clf.tfidf.meta = {
        "name": "tfidf-v2",
        "task": "polarity",
        "trained_with": f"PerSent {__version__}",
        "date": date.today().isoformat(),
        "training_data": "RadeAI Digikala (Comments & Products), cleaned with scripts/build_polarity_dataset.py",
        "n_train": len(train),
        "config": {**{k: list(v) if isinstance(v, tuple) else v for k, v in CONFIG.items()}, "C": C},
        "metrics": metrics,
    }
    clf.save(a.output)
    print("saved", a.output, f"{a.output.stat().st_size / 1e6:.1f} MB")
    print(json.dumps(clf.tfidf.meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
