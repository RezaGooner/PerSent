"""Convert PerSent <= 1.3.x pickle-based models to the pickle-free formats used since 1.4.0.

Legacy files (joblib/pickle):
    classifier.joblib                 sklearn LogisticRegression
    word2vec.model                    gensim Word2Vec
    weighted_sentiment_model.joblib   dict with emotion_map / keywords / word_weights

New files:
    comment_model.npz                 embeddings + linear classifier (numpy, allow_pickle=False)
    emotion_lexicon.json              emotion lexicon (plain JSON)

Usage:
    python scripts/convert_legacy_models.py LEGACY_DIR OUTPUT_DIR

Only run this on model files you trust: loading pickle files can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from gensim.models import Word2Vec

LEGACY_LABELS = {0: "not_recommended", 1: "recommended", 2: "no_idea"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("legacy_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from PerSent._models import EmotionLexicon, LinearClassifier, WordEmbeddings, save_comment_model

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clf_path = args.legacy_dir / "classifier.joblib"
    w2v_path = args.legacy_dir / "word2vec.model"
    if clf_path.exists() and w2v_path.exists():
        clf = joblib.load(clf_path)
        w2v = Word2Vec.load(str(w2v_path))
        embeddings = WordEmbeddings(list(w2v.wv.index_to_key), np.asarray(w2v.wv.vectors, dtype=np.float32))
        labels = [LEGACY_LABELS[int(c)] for c in clf.classes_]
        classifier = LinearClassifier(clf.coef_, clf.intercept_, labels)
        out = args.output_dir / "comment_model.npz"
        save_comment_model(out, embeddings, classifier)
        print(f"wrote {out}")

    emo_path = args.legacy_dir / "weighted_sentiment_model.joblib"
    if emo_path.exists():
        data = joblib.load(emo_path)
        lexicon = EmotionLexicon(data["emotion_map"], data["keywords"], data["word_weights"])
        out = args.output_dir / "emotion_lexicon.json"
        lexicon.save(out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
