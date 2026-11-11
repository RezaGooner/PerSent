"""Build a clean, balanced polarity dataset for PerSent from the Digikala comments dump.

Source: RadeAI "Digikala (Comments & Products)" (MIT, per its Hugging Face card),
the same file as Sentiment-Survey-Analyzer/Dataset/big_train.

Cleaning
  * body only (titles often repeat the label, e.g. "پیشنهاد نمیشود")
  * whitespace normalised, empty / 1-character bodies dropped
  * label must agree with the star rating:
        recommended      rate > 3.5
        not_recommended  0 < rate <= 2.5
        no_idea          1.5 < rate <= 3.5
  * exact duplicates removed; texts that occur with different labels removed
Splits are by product_id (no product appears in two splits) and balanced per class.

Usage: python build_polarity_dataset.py BIG_TRAIN_CSV OUT_DIR [--per-class 100000]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

WS = re.compile(r"\s+")
RULES = {
    "recommended": lambda r: r > 3.5,
    "not_recommended": lambda r: (r > 0) & (r <= 2.5),
    "no_idea": lambda r: (r > 1.5) & (r <= 3.5),
}
# rough keep-rates per chunk so the big class does not blow up memory
KEEP = {"recommended": 0.12, "not_recommended": 1.0, "no_idea": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--per-class", type=int, default=100_000, help="training rows per class")
    ap.add_argument("--eval-per-class", type=int, default=10_000, help="validation/test rows per class")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    parts, stats = [], {"read": 0}
    cols = ["body", "rate", "recommendation_status", "product_id", "is_buyer"]
    for ch in pd.read_csv(a.src, usecols=cols, dtype={"body": str, "recommendation_status": str}, chunksize=500_000):
        stats["read"] += len(ch)
        ch = ch.dropna(subset=["body", "recommendation_status", "rate", "product_id"])
        ch = ch[ch.rate <= 5]
        ch["body"] = ch.body.str.replace(WS, " ", regex=True).str.strip()
        ch = ch[ch.body.str.len() > 1]
        keep = np.zeros(len(ch), dtype=bool)
        for label, rule in RULES.items():
            m = (ch.recommendation_status == label).to_numpy() & rule(ch.rate).to_numpy()
            keep |= m & (rng.random(len(ch)) < KEEP[label])
        parts.append(ch[keep])
    df = pd.concat(parts, ignore_index=True)
    stats["consistent (sampled)"] = len(df)

    conflict = df.groupby("body").recommendation_status.nunique()
    df = df[~df.body.isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("body")
    stats["deduplicated"] = len(df)
    print(stats)
    print(df.recommendation_status.value_counts().to_dict())

    # product-disjoint split: assign each product to train/valid/test
    products = df.product_id.unique()
    rng.shuffle(products)
    n = len(products)
    split_of = pd.Series("train", index=products)
    split_of.iloc[: int(0.1 * n)] = "test"
    split_of.iloc[int(0.1 * n) : int(0.2 * n)] = "valid"
    df["split"] = df.product_id.map(split_of)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, per_class in (("train", a.per_class), ("valid", a.eval_per_class), ("test", a.eval_per_class)):
        sub = df[df.split == split]
        frames = []
        for label in RULES:
            g = sub[sub.recommendation_status == label]
            if len(g) < per_class:
                print(f"warning: {split}/{label} has only {len(g)} rows")
            frames.append(g.sample(min(per_class, len(g)), random_state=a.seed))
        res = pd.concat(frames).sample(frac=1, random_state=a.seed)
        res = res[["body", "recommendation_status", "rate", "product_id"]]
        res.to_csv(out / f"{split}.csv", index=False)
        print(
            split, len(res), res.recommendation_status.value_counts().to_dict(), "products:", res.product_id.nunique()
        )


if __name__ == "__main__":
    main()
