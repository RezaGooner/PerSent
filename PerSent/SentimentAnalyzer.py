# -*- coding: utf-8 -*-
"""
Persian Weighted Sentiment Analyzer Library
A comprehensive sentiment analysis tool for Persian text with word weighting support.
"""

import re
import csv
import os
import joblib
from collections import defaultdict
from hazm import Stemmer, Lemmatizer, Normalizer, word_tokenize, stopwords_list
from tqdm import tqdm
import pandas as pd

class SentimentAnalyzer:
    """A Persian sentiment analyzer that supports word weighting"""
    
    def __init__(self, model_dir='PerSent/model'):
        """
        Initialize the weighted sentiment analyzer
        
        Args:
            model_dir (str): Directory path for saving/loading models
        """
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()
        self.stopwords = set(stopwords_list())
        self.model_dir = model_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Emotion mapping dictionary
        self.emotion_map = {
            'شادی': 1,  # Happiness
            'غم': 2,    # Sadness
            'ترس': 3,   # Fear
            'عصبانیت': 4,  # Anger
            'تنفر': 5,  # Disgust
            'شگفتی': 6,  # Surprise
            'آرامش': 7   # Calm
        }
        
        self.keywords = defaultdict(list)
        self.word_weights = defaultdict(dict)  # Stores word weights for each emotion
        self.model_loaded = False
    
    def _normalize_text(self, text):
        """Normalize Persian text by removing half-spaces and punctuation"""
        text = re.sub(r'[\u200c\s]+', ' ', text).strip()
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = re.sub(r'\d+', '', text)
        return text.lower()
    
    def _preprocess_text(self, text):
        """Preprocess text into stemmed tokens"""
        text = self._normalize_text(text)
        tokens = word_tokenize(text)
        return [self.stemmer.stem(t) for t in tokens if t not in self.stopwords and len(t) > 1]
    
    def _generate_word_forms(self, word):
        """Generate all possible forms of a word (stem, lemma, etc.)"""
        stem = self.stemmer.stem(word)
        lemma = self.lemmatizer.lemmatize(word)
        
        forms = {
            word,
            stem,
            lemma,
            self.stemmer.stem(stem),
            self.lemmatizer.lemmatize(stem),
            self.stemmer.stem(lemma),
            self.lemmatizer.lemmatize(lemma)
        }
        return [f for f in forms if f and len(f) > 1]
    
    def load_weighted_lexicon(self, csv_file, word_col=0, emotion_col=1, weight_col=2):
        """
        Load a weighted lexicon from CSV file
        
        Args:
            csv_file (str): Path to CSV file
            word_col (int/str): Column index/name for words (default: 0)
            emotion_col (int/str): Column index/name for emotions (default: 1)
            weight_col (int/str): Column index/name for weights (default: 2)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Lexicon file {csv_file} not found")
            
        try:
            with open(csv_file, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) < max(word_col, emotion_col, weight_col) + 1:
                        continue
                        
                    # Handle both column names and indices
                    word = str(row[word_col]).strip() if isinstance(word_col, int) else row[word_col]
                    emotion = str(row[emotion_col]).strip() if isinstance(emotion_col, int) else row[emotion_col]
                    
                    try:
                        weight = float(row[weight_col]) if isinstance(weight_col, int) else float(row[weight_col])
                    except (ValueError, IndexError):
                        weight = 1.0  # Default weight if invalid
                    
                    word = self._normalize_text(word)
                    forms = self._generate_word_forms(word)
                    
                    if emotion in self.emotion_map:
                        numeric_emotion = self.emotion_map[emotion]
                        self.keywords[numeric_emotion].extend(forms)
                        
                        # Store weights for each word form
                        for form in forms:
                            current_weight = self.word_weights.get(form, {}).get(numeric_emotion, 0)
                            if weight > current_weight:
                                self.word_weights.setdefault(form, {})[numeric_emotion] = weight
            
            # Remove duplicates
            for emotion in self.keywords:
                self.keywords[emotion] = list(set(self.keywords[emotion]))
                
            self.model_loaded = True
                
        except Exception as e:
            raise ValueError(f"Error processing lexicon file: {str(e)}")
    
    def train_from_csv(self, train_csv, text_col='text', emotion_col='emotion', weight_col='weight'):
        """
        Train model from a weighted CSV dataset
        
        Args:
            train_csv (str): Path to training CSV file
            text_col (str/int): Column name/index for text (default: 'text')
            emotion_col (str/int): Column name/index for emotions (default: 'emotion')
            weight_col (str/int): Column name/index for weights (default: 'weight')
            
        Returns:
            dict: Training result with status and message
            
        Raises:
            FileNotFoundError: If training file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Training file {train_csv} not found")
            
        try:
            df = pd.read_csv(train_csv)
            
            # Handle column names/indices
            text_col = df.columns[text_col] if isinstance(text_col, int) else text_col
            emotion_col = df.columns[emotion_col] if isinstance(emotion_col, int) else emotion_col
            weight_col = df.columns[weight_col] if isinstance(weight_col, int) else weight_col
            
            # Check required columns
            missing_cols = [col for col in [text_col, emotion_col, weight_col] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
            
            # Train model with weights
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Training model"):
                text = str(row[text_col])
                emotion = str(row[emotion_col])
                weight = float(row[weight_col])
                
                words = self._preprocess_text(text)
                forms = set()
                for word in words:
                    forms.update(self._generate_word_forms(word))
                
                if emotion in self.emotion_map:
                    numeric_emotion = self.emotion_map[emotion]
                    self.keywords[numeric_emotion].extend(forms)
                    
                    # Update weights
                    for form in forms:
                        current_weight = self.word_weights.get(form, {}).get(numeric_emotion, 0)
                        if weight > current_weight:
                            self.word_weights.setdefault(form, {})[numeric_emotion] = weight
            
            # Save trained model
            self.save_model()
            self.model_loaded = True
            
            return {"status": "success", "message": "Model trained successfully"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def save_model(self, model_name='weighted_sentiment_model'):
        """
        Save the current model to disk
        
        Args:
            model_name (str): Base name for model files (without extension)
        """
        model_path = os.path.join(self.model_dir, f'{model_name}.joblib')
        joblib.dump({
            'emotion_map': self.emotion_map,
            'keywords': dict(self.keywords),
            'word_weights': dict(self.word_weights)
        }, model_path)
    
    def load_model(self, model_name='weighted_sentiment_model'):
        """
        Load a saved model from disk
        
        Args:
            model_name (str): Base name for model files (without extension)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted
        """
        model_path = os.path.join(self.model_dir, f'{model_name}.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
            
        try:
            data = joblib.load(model_path)
            self.emotion_map = data['emotion_map']
            self.keywords = defaultdict(list, data['keywords'])
            self.word_weights = defaultdict(dict, data['word_weights'])
            self.model_loaded = True
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def analyze_text(self, text):
        """
        Analyze sentiment of input text with weighted scoring
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Emotion percentages with weights
            
        Raises:
            Exception: If model is not loaded
        """
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load or train a model first.")
            
        words = self._preprocess_text(text)
        emotion_scores = defaultdict(float)
        total_score = 0.0
        
        for word in words:
            forms = self._generate_word_forms(word)
            
            for form in forms:
                if form in self.word_weights:
                    for emotion_id, weight in self.word_weights[form].items():
                        emotion_scores[emotion_id] += weight
                        total_score += weight
        
        # Calculate weighted percentages
        results = {}
        reverse_map = {v: k for k, v in self.emotion_map.items()}
        
        if total_score > 0:
            for emotion_id in self.emotion_map.values():
                score = emotion_scores.get(emotion_id, 0)
                results[reverse_map[emotion_id]] = round(score / total_score * 100, 2)
        
        return results or {emotion: 0.0 for emotion in self.emotion_map}
    
    def analyze_csv(self, input_csv, output_csv, text_col='text', output_col='sentiment_analysis'):
        """
        Analyze sentiment for a CSV file and save results
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str): Path to save output CSV
            text_col (str/int): Column name/index containing text (default: 'text')
            output_col (str): Column name for output results (default: 'sentiment_analysis')
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            Exception: If model is not loaded
        """
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load or train a model first.")
            
        try:
            df = pd.read_csv(input_csv)
            
            # Handle column name/index
            text_col = df.columns[text_col] if isinstance(text_col, int) else text_col
            
            if text_col not in df.columns:
                raise ValueError(f"Column '{text_col}' not found in CSV file")
            
            tqdm.pandas(desc="Analyzing sentiments")
            df[output_col] = df[text_col].progress_apply(
                lambda x: self.analyze_text(str(x)))
            
            # Save results
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            return True
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

