"""Deprecated 1.x interface for emotion analysis — use :class:`PerSent.EmotionAnalyzer`.

Kept so that code written for PerSent <= 1.4 keeps working unchanged. It will be
removed in PerSent 2.0.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._models import EmotionLexicon
from ._text import emotion_tokens, normalize_emotion_text, word_forms
from ._utils import logger, warn_deprecated
from .emotion import DEFAULT_MODEL_NAME, EMOTION_MAP, EmotionAnalyzer
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError

__all__ = ["DEFAULT_MODEL_NAME", "EMOTION_MAP", "SentimentAnalyzer"]


class SentimentAnalyzer:
    """Deprecated: use :class:`PerSent.EmotionAnalyzer`.

    ==================================  ===================================================
    1.x                                 1.5+
    ==================================  ===================================================
    ``SentimentAnalyzer()``             ``EmotionAnalyzer()``
    ``loadModel()``                     ``EmotionAnalyzer.pretrained()`` / ``.load(path)``
    ``saveModel(name)``                 ``save(path)``
    ``train(csv)``                      ``fit_csv(csv)``
    ``loadLex(csv)``                    ``add_lexicon(csv)``
    ``analyzeText(t)``                  ``analyze(t).scores``
    ``analyzeCSV(i, o)``                ``analyze_csv(i, o)``
    ==================================  ===================================================
    """

    def __init__(self, model_dir: str | os.PathLike | None = None) -> None:
        warn_deprecated("SentimentAnalyzer", "PerSent.EmotionAnalyzer")
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self._emo = EmotionAnalyzer(EmotionLexicon(dict(EMOTION_MAP)))
        self.model_loaded = False

    # 1.4 exposed these attributes
    @property
    def model(self) -> EmotionAnalyzer:
        """The underlying :class:`EmotionAnalyzer`."""
        return self._emo

    @property
    def emotion_map(self) -> dict[str, int]:
        return self._emo.lexicon.emotion_map

    @emotion_map.setter
    def emotion_map(self, value: dict[str, int]) -> None:
        self._emo.lexicon.emotion_map = dict(value)

    @property
    def keywords(self):
        return self._emo.lexicon.keywords

    @keywords.setter
    def keywords(self, value) -> None:
        self._emo.lexicon.keywords = value

    @property
    def word_weights(self):
        return self._emo.lexicon.word_weights

    @word_weights.setter
    def word_weights(self, value) -> None:
        self._emo.lexicon.word_weights = value

    def _normalize_text(self, text: object) -> str:
        return normalize_emotion_text(text)

    def _preprocess_text(self, text: object) -> list[str]:
        return emotion_tokens(text)

    def _generate_word_forms(self, word: str) -> list[str]:
        return word_forms(word)

    def loadLex(
        self,
        csv_file: str | os.PathLike,
        word_col: int | str = 0,
        emotion_col: int | str = 1,
        weight_col: int | str = 2,
    ) -> None:
        self._emo.add_lexicon(csv_file, word_col, emotion_col, weight_col)
        self.model_loaded = True

    def train(
        self,
        train_csv: str | os.PathLike,
        text_col: int | str = "text",
        emotion_col: int | str | None = None,
        weight_col: int | str = "weight",
        reset: bool = False,
    ) -> dict:
        """Returns ``{"status": "success" | "error", "message": ...}``."""
        try:
            self._emo.fit_csv(train_csv, text_col, emotion_col, weight_col, reset=reset, show_progress=True)
            self.model_loaded = True
            if self.model_dir is not None:
                self.saveModel()
            return {"status": "success", "message": "Model trained successfully"}
        except FileNotFoundError as exc:
            return {"status": "error", "message": f"File not found: {exc}"}
        except (DataValidationError, ValueError) as exc:
            return {"status": "error", "message": f"Data validation error: {exc}"}
        except Exception as exc:
            return {"status": "error", "message": f"Training failed: {exc}"}

    def saveModel(self, model_name: str = DEFAULT_MODEL_NAME, model_dir: str | os.PathLike | None = None) -> Path:
        target = Path(model_dir) if model_dir is not None else self.model_dir
        if target is None:
            raise ValueError("No model_dir given: use SentimentAnalyzer(model_dir=...) or saveModel(model_dir=...)")
        return self._emo.save(target / f"{model_name}.json")

    def loadModel(self, model_name: str = DEFAULT_MODEL_NAME, model_dir: str | os.PathLike | None = None) -> None:
        """Load ``<model_dir>/<model_name>.json`` or, for the default name, the bundled model."""
        directory = Path(model_dir) if model_dir is not None else self.model_dir
        if directory is not None:
            path = directory / f"{model_name}.json"
            legacy = directory / f"{model_name}.joblib"
            if path.exists():
                self._emo = EmotionAnalyzer.load(path)
                self.model_loaded = True
                return
            if legacy.exists():
                self._emo = EmotionAnalyzer._load_legacy(legacy, stacklevel=4)
                self.model_loaded = True
                return
        if model_name != DEFAULT_MODEL_NAME:
            where = directory if directory is not None else "(no model_dir set)"
            raise ModelFileError(f"Model '{model_name}' not found in {where}")
        if directory is not None:
            logger.warning("No model found in %s; loading the bundled pre-trained model.", directory)
        self._emo = EmotionAnalyzer.pretrained()
        self.model_loaded = True

    def analyzeText(self, text: str) -> dict[str, float]:
        if not self.model_loaded:
            raise ModelNotLoadedError("Model not loaded. Please load or train a model first.")
        return self._emo.analyze(text).scores

    def analyzeCSV(
        self,
        input_csv: str | os.PathLike,
        output_csv: str | os.PathLike,
        text_col: int | str = "text",
        output_col: str = "sentiment_analysis",
    ) -> bool:
        """Returns ``True`` on success, ``False`` on failure (logged)."""
        try:
            if not self.model_loaded:
                raise ModelNotLoadedError("Model not loaded. Please load or train a model first.")
            self._emo.analyze_csv(input_csv, output_csv, text_column=text_col, output_column=output_col, expand=False)
            return True
        except FileNotFoundError as exc:
            logger.error("Input file not found - %s", exc)
        except (DataValidationError, ValueError) as exc:
            logger.error("Invalid column specification - %s", exc)
        except Exception as exc:
            logger.error("Error during CSV analysis: %s", exc)
        return False
