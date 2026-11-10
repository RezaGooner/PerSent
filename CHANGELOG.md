# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.4.0] - Unreleased

### Added
- `pyproject.toml` packaging (hatchling) with a single version source (`src/PerSent/_version.py`).
- Support for Python 3.12 and 3.13 (requires Python ≥ 3.10).
- Test suite (pytest), linting (ruff) and GitHub Actions CI on Linux, Windows and macOS.
- Release workflow that publishes to PyPI with Trusted Publishing.
- `CommentAnalyzer.analyzeTexts()` for lists of texts.
- `CommentAnalyzer.train()` parameters `random_state`, `text_column`, `label_column`, `workers`; the split is now stratified.
- `SentimentAnalyzer.train(reset=...)` to start from an empty model.
- Exception hierarchy: `PerSentError`, `ModelNotLoadedError`, `ModelFileError`, `DataValidationError`
  (all subclasses of `Exception`).
- Type hints (`py.typed`) and `scripts/convert_legacy_models.py`.
- Vendored hazm 0.9's suffix stemmer (MIT), so preprocessing — and therefore predictions — stay the same with
  hazm 0.12 on Python 3.12+ (whose new stemmer changed ~10 % of predictions).

### Changed
- Pre-trained models are stored in pickle-free formats (`comment_model.npz`, `emotion_lexicon.json`).
  Predictions are identical to 1.3.3; the wheel shrank from ~27 MB to ~14 MB.
- `model_dir` now defaults to `None`: `loadModel()` loads the bundled model and `train()` only saves when a
  `model_dir` is set. Legacy 1.3.x model files in an explicit `model_dir` still load (with a `DeprecationWarning`).
- Status messages use `logging` (logger `"PerSent"`) instead of `print`.
- Removed the `numpy<2` / `scipy<1.14` pins that made the package uninstallable on Python 3.12+.
- Repository layout moved to `src/`.

### Fixed
- **Security:** `loadModel()` no longer loads (pickle) files from `./PerSent/model` in the current working
  directory, and creating an analyzer no longer creates that folder.
- `SentimentAnalyzer.train()` failed with its default arguments on the bundled dataset (`emotion_col` default).
- `CommentAnalyzer` deleted all upper-case Latin letters ("iPhone" → "i hone") due to a regex range bug.
- `CommentAnalyzer.train(vector_size=...)` crashed for any size other than 100.
- Empty labels in the training CSV crashed training instead of being treated as `no_idea`.
- `SentimentAnalyzer.loadLex()` crashed when columns were given by name.
- Empty or digit-only texts were classified as `recommended`; they now return `no_idea`.
- Missing values (NaN) in CSV text columns were analyzed as the word "nan".
- With hazm >= 0.11, every text re-created hazm's `WordTokenizer` (~0.3 s per text); one instance is now reused.

### Removed
- `setup.py`, `setup.cfg`, `MANIFEST.in`, `requirements.txt`, committed `build/` output and `PKG-INFO`.
- Draft JOSS paper files (`paper.md`, `paper.bib`).

## [1.3.3] - 2025-08-22
- Last release with the original `setup.py` layout and pickle-based models.

[1.4.0]: https://github.com/RezaGooner/PerSent/compare/088b919...HEAD
[1.3.3]: https://pypi.org/project/PerSent/1.3.3/
