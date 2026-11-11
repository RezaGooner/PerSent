"""Merge public Persian emotion datasets into PerSent's training format.

Output columns: text, sentiment (Persian emotion name), weight, source, license
  ArmanEmo         (CC BY-NC-SA 4.0)  https://github.com/Arman-Rayan-Sharif/arman-text-emotion
  ShortPersianEmo  (GPL-3.0)          https://github.com/vkiani/ShortPersianEmo
  EmoPars          (no license given) https://github.com/nazaninsbr/Persian-Emotion-Detection

"OTHER" samples are dropped (PerSent has no such class). None of the sources has a
calmness (آرامش) class. EmoPars is multi-annotator (0-5 votes per emotion): a row is
kept when one emotion has >= 3 votes and strictly more than every other emotion;
weight = votes / 5. Official test splits are kept; EmoPars gets a random 10 % test
split. Texts that appear in any test split are removed from all training splits.

Usage: python build_emotion_dataset.py DATASETS_DIR OUT_DIR
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAP = {
    "HAPPY": "شادی", "Happiness": "شادی",
    "SAD": "غم", "Sadness": "غم",
    "FEAR": "ترس", "Fear": "ترس",
    "ANGRY": "عصبانیت", "Anger": "عصبانیت",
    "HATE": "تنفر", "Hatred": "تنفر",
    "SURPRISE": "شگفتی", "Wonder": "شگفتی",
}  # fmt: skip
WS = re.compile(r"\s+")


def clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.strip('"').str.replace(WS, " ", regex=True).str.strip()


def arman(root: Path) -> pd.DataFrame:
    out = []
    for split in ("train", "test"):
        d = pd.read_csv(root / "arman-text-emotion/dataset" / f"{split}.tsv", sep="\t", header=None,
                        quoting=csv.QUOTE_NONE, names=["text", "label"])  # fmt: skip
        d["split"] = split
        out.append(d)
    d = pd.concat(out)
    d["weight"], d["source"], d["license"] = 1.0, "ArmanEmo", "CC BY-NC-SA 4.0"
    return d


def short_persian_emo(root: Path) -> pd.DataFrame:
    out = []
    for split in ("train", "test"):
        d = pd.read_excel(root / "ShortPersianEmo/Data/ShortPersianEmo" / f"{split}_fa.xlsx", header=None,
                          names=["text", "label"])  # fmt: skip
        d["split"] = split
        out.append(d)
    d = pd.concat(out)
    d["weight"], d["source"], d["license"] = 1.0, "ShortPersianEmo", "GPL-3.0"
    return d


def emopars(root: Path) -> pd.DataFrame:
    d = pd.read_csv(root / "Persian-Emotion-Detection/dataset.csv")
    emos = ["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"]
    votes = d[emos].to_numpy()
    top = votes.max(axis=1)
    unique_top = (votes == top[:, None]).sum(axis=1) == 1
    keep = (top >= 3) & unique_top
    d = d[keep].copy()
    d["label"] = [emos[i] for i in votes[keep].argmax(axis=1)]
    d["weight"] = top[keep] / 5
    rng = np.random.default_rng(42)
    d["split"] = np.where(rng.random(len(d)) < 0.1, "test", "train")
    d["source"], d["license"] = "EmoPars", "unspecified"
    return d[["text", "label", "weight", "split", "source", "license"]]


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = pd.concat([arman(root), short_persian_emo(root), emopars(root)], ignore_index=True)
    df["text"] = clean(df.text)
    df = df[df.text.str.len() > 1]
    print("raw:", df.groupby(["source", "split"]).size().to_dict())
    df["sentiment"] = df.label.map(MAP)
    df = df.dropna(subset=["sentiment"])  # drops OTHER

    test_texts = set(df.loc[df.split == "test", "text"])
    train = df[(df.split == "train") & ~df.text.isin(test_texts)]
    train = train.drop_duplicates(["text", "sentiment"]).drop_duplicates("text", keep=False)
    test = df[df.split == "test"].drop_duplicates("text", keep=False)

    out.mkdir(parents=True, exist_ok=True)
    cols = ["text", "sentiment", "weight", "source", "license"]
    for name, part in (("train", train), ("test", test)):
        part = part.sample(frac=1, random_state=42)[cols]
        part.to_csv(out / f"{name}.csv", index=False)
        print(name, len(part), part.sentiment.value_counts().to_dict(), part.source.value_counts().to_dict())
        for src, g in part.groupby("source"):
            g.to_csv(out / f"{name}_{src}.csv", index=False)


if __name__ == "__main__":
    main()
