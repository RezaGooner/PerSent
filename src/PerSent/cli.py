"""Command-line interface: ``persent`` (or ``python -m PerSent``).

Examples::

    persent polarity "کیفیت عالی داشت" --proba
    echo "خیلی خوشحالم" | persent emotion --json
    persent csv comments.csv -o results.csv --task polarity --column نظرات --summary summary.csv
    persent train polarity train.csv -o my_model/
    persent evaluate test.csv --model my_model/
    persent polarity "خوب بود" -m word2vec-v1          # the PerSent <= 1.5 model
    persent train emotion emotions.csv -o emo.npz --method classifier
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Sequence

from ._version import __version__

MODEL_HELP = "bundled model name (e.g. tfidf-v2, word2vec-v1, lexicon-v1) or a model directory/file"


def _configure_stdio() -> None:
    # Persian text on Windows consoles (cp1252, cp720, ...) would otherwise raise UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8")


def _texts(args: argparse.Namespace) -> list[str]:
    if args.text:
        return list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8-sig") as fh:
            return [line.rstrip("\n") for line in fh if line.strip()]
    if sys.stdin.isatty():
        raise SystemExit("error: give one or more texts, --file FILE, or pipe text via stdin")
    return [line.rstrip("\n") for line in sys.stdin if line.strip()]


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _polarity_model(path: str | None):
    from .polarity import PolarityClassifier

    if not path or path in PolarityClassifier.available_models():
        return PolarityClassifier.pretrained(path)
    return PolarityClassifier.load(path)


def _emotion_model(path: str | None):
    from .emotion import EmotionAnalyzer

    if not path or path in EmotionAnalyzer.available_models():
        return EmotionAnalyzer.pretrained(path)
    return EmotionAnalyzer.load(path)


def cmd_polarity(args: argparse.Namespace) -> int:
    clf = _polarity_model(args.model)
    for result in clf.analyze_many(_texts(args)):
        if args.json:
            print(_dump(result.to_dict()))
        elif args.proba:
            scores = "  ".join(f"{k}={v:.3f}" for k, v in sorted(result.scores.items(), key=lambda kv: -kv[1]))
            print(f"{result.label}\t{scores}\t{result.text}")
        else:
            print(f"{result.label}\t{result.text}")
    return 0


def cmd_emotion(args: argparse.Namespace) -> int:
    emo = _emotion_model(args.model)
    for result in emo.analyze_many(_texts(args)):
        if args.json:
            print(_dump(result.to_dict(lang=args.lang)))
            continue
        scores = result.scores_en() if args.lang == "en" else result.scores
        dominant = (result.dominant_en if args.lang == "en" else result.dominant) or "-"
        top = "  ".join(f"{k}={v:.1f}%" for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v > 0)
        print(f"{dominant}\t{top or '-'}\t{result.text}")
    return 0


def cmd_csv(args: argparse.Namespace) -> int:
    column: int | str = int(args.column) if args.column.lstrip("-").isdigit() else args.column
    if args.task == "polarity":
        df = _polarity_model(args.model).analyze_csv(
            args.input,
            args.output,
            text_column=column,
            include_scores=args.scores,
            summary_csv=args.summary,
            show_progress=not args.quiet,
        )
    else:
        if args.summary:
            raise SystemExit("error: --summary is only available for --task polarity")
        df = _emotion_model(args.model).analyze_csv(
            args.input, args.output, text_column=column, lang=args.lang, show_progress=not args.quiet
        )
    if not args.quiet:
        print(f"Analyzed {len(df)} rows -> {args.output}", file=sys.stderr)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if args.task == "polarity":
        from .polarity import PolarityClassifier

        method = args.method or "tfidf"
        if method not in ("tfidf", "word2vec"):
            raise SystemExit("error: --method for polarity must be tfidf or word2vec")
        options = (
            {"vector_size": args.vector_size, "window": args.window, "epochs": args.epochs}
            if method == "word2vec"
            else {}
        )
        clf = PolarityClassifier()
        report = clf.fit_csv(
            args.data,
            text_column=args.text_column or "body",
            label_column=args.label_column or "recommendation_status",
            method=method,
            test_size=args.test_size,
            random_state=args.seed,
            show_progress=not args.quiet,
            **options,
        )
        path = clf.save(args.output)
        print(report, file=sys.stderr)
    else:
        from .emotion import EmotionAnalyzer

        method = args.method or "lexicon"
        if method == "classifier":
            emo = EmotionAnalyzer()
            report = emo.fit_classifier_csv(
                args.data,
                text_column=args.text_column or "text",
                emotion_column=args.label_column,
                test_size=args.test_size,
                random_state=args.seed,
            )
            print(report, file=sys.stderr)
        elif method == "lexicon":
            emo = EmotionAnalyzer.pretrained() if args.extend else EmotionAnalyzer()
            used = emo.fit_csv(
                args.data,
                text_column=args.text_column or "text",
                emotion_column=args.label_column,
                show_progress=not args.quiet,
            )
            print(f"Learned from {used} rows", file=sys.stderr)
        else:
            raise SystemExit("error: --method for emotion must be lexicon or classifier")
        path = emo.save(args.output)
    print(f"Model saved to {path}", file=sys.stderr)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.task == "emotion":
        report = _emotion_model(args.model).evaluate_csv(
            args.data, text_column=args.text_column or "text", emotion_column=args.label_column
        )
    else:
        report = _polarity_model(args.model).evaluate_csv(
            args.data,
            text_column=args.text_column or "body",
            label_column=args.label_column or "recommendation_status",
        )
    if args.json:
        print(_dump(report.to_dict()))
        return 0
    print(report)
    print(f"{'label':<18}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label, m in report.per_label.items():
        print(f"{label:<18}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="persent", description="Persian sentiment and emotion analysis.")
    parser.add_argument("--version", action="version", version=f"PerSent {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show log messages")
    sub = parser.add_subparsers(dest="command", required=True)

    def text_inputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("text", nargs="*", help="text(s) to analyze; omit to read lines from --file or stdin")
        p.add_argument("-f", "--file", help="UTF-8 text file with one text per line")
        p.add_argument("-m", "--model", help=MODEL_HELP)
        p.add_argument("--json", action="store_true", help="one JSON object per line")

    p = sub.add_parser("polarity", help="recommended / not_recommended / no_idea")
    text_inputs(p)
    p.add_argument("--proba", action="store_true", help="also print label probabilities")
    p.set_defaults(func=cmd_polarity)

    p = sub.add_parser("emotion", help="seven-emotion scores")
    text_inputs(p)
    p.add_argument("--lang", choices=("fa", "en"), default="fa", help="emotion names in output")
    p.set_defaults(func=cmd_emotion)

    p = sub.add_parser("csv", help="analyze a column of a CSV file")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-t", "--task", choices=("polarity", "emotion"), default="polarity")
    p.add_argument("-c", "--column", default="0", help="column name or 0-based index (default: 0)")
    p.add_argument("-m", "--model", help=MODEL_HELP)
    p.add_argument("--scores", action="store_true", help="polarity: add one probability column per label")
    p.add_argument("--summary", help="polarity: also write a label summary CSV")
    p.add_argument("--lang", choices=("fa", "en"), default="fa", help="emotion: column names")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_csv)

    p = sub.add_parser("train", help="train a model from a CSV file")
    p.add_argument("task", choices=("polarity", "emotion"))
    p.add_argument("data", help="training CSV")
    p.add_argument("-o", "--output", required=True, help="directory or file to save the model to")
    p.add_argument(
        "--method",
        help="polarity: tfidf (default) or word2vec; emotion: lexicon (default) or classifier",
    )
    p.add_argument("--text-column", help="default: body (polarity) / text (emotion)")
    p.add_argument("--label-column", help="default: recommendation_status (polarity) / sentiment (emotion)")
    p.add_argument("--vector-size", type=int, default=100, help="word2vec only")
    p.add_argument("--window", type=int, default=5, help="word2vec only")
    p.add_argument("--epochs", type=int, default=5, help="word2vec only")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int)
    p.add_argument("--extend", action="store_true", help="emotion: start from the bundled lexicon")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="accuracy / F1 of a model on a labelled CSV")
    p.add_argument("data")
    p.add_argument("-t", "--task", choices=("polarity", "emotion"), default="polarity")
    p.add_argument("-m", "--model", help=MODEL_HELP)
    p.add_argument("--text-column", help="default: body (polarity) / text (emotion)")
    p.add_argument("--label-column", help="default: recommendation_status (polarity) / sentiment (emotion)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    from .exceptions import PerSentError

    try:
        return args.func(args)
    except (PerSentError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
