from .CommentAnalyzer import CommentAnalyzer
from .SentimentAnalyzer import SentimentAnalyzer
from pathlib import Path
import os

MODEL_DIR = Path(__file__).parent / "model"
os.makedirs(MODEL_DIR, exist_ok=True)

__version__ = "1.1.0"
__all__ = ['CommentAnalyzer', 'SentimentAnalyzer', 'MODEL_DIR']
