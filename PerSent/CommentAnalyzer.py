#import necessary library
import pandas as pd
from hazm import Normalizer, word_tokenize, Stemmer, stopwords_list
import re
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import joblib
import importlib.resources


class CommentAnalyzer:
    def __init__(self, model_dir='PerSent/model'):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.stopwords = set(stopwords_list())
        self.model_dir = model_dir
        self.vectorizer = None
        self.classifier = None
        
        # make /model Directory if not exist
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _preprocess_text(self, text):
        """PreProcess Persian Text"""
        try:
            # Normalizing
            text = self.normalizer.normalize(str(text))
            
            # remove number and sign
            text = re.sub(r'[!()-\[\]{};:\'",؟<>./?@#$%^&*_~۰-۹\d]+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # tokenize and stemming
            tokens = word_tokenize(text)
            processed_tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token not in self.stopwords and len(token) > 1
            ]
            
            return processed_tokens
        except Exception as e:
            raise Exception(f"Text preprocessing failed: {str(e)}")
    
    def _sentence_vector(self, sentence, model):
        """convert sentences to vector by word2vec model"""
        try:
            vectors = []
            for word in sentence:
                try:
                    vectors.append(model.wv[word])
                except KeyError:
                    vectors.append(np.zeros(100))
            return np.mean(vectors, axis=0) if vectors else np.zeros(100)
        except Exception as e:
            raise Exception(f"Sentence vectorization failed: {str(e)}")
    
    def train(self, train_csv, test_size=0.2, vector_size=100, window=5):
        """Train model"""
        try:
            # read data
            if not os.path.exists(train_csv):
                raise FileNotFoundError(f"Training file {train_csv} not found")
                
            df = pd.read_csv(train_csv)
            
            if 'body' not in df.columns or 'recommendation_status' not in df.columns:
                raise ValueError("CSV file must contain 'body' and 'recommendation_status' columns")
            
            df['tokens'] = df['body'].apply(self._preprocess_text)
            
            # train Word2Vec model
            self.vectorizer = Word2Vec(
                sentences=df['tokens'],
                vector_size=vector_size,
                window=window,
                min_count=1,
                workers=4
            )
            
            # convert sentences to vector
            X = np.array([self._sentence_vector(s, self.vectorizer) for s in df['tokens']])
            y = df['recommendation_status'].map({
                "no_idea": 2,
                "recommended": 1,
                "not_recommended": 0
            }).values
            
            # make train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            self.classifier = LogisticRegression(max_iter=1000)
            self.classifier.fit(X_train, y_train)
            
            # save model
            self.saveModel()
            
            # evaluation
            accuracy = self.classifier.score(X_test, y_test)
            return accuracy
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Training file error: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Data validation error: {str(e)}")
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def analyzeText(self, text):
        """Predict text sentiment"""
        try:
            if not self.classifier or not self.vectorizer:
                raise Exception("Model not trained! Call train() first or load a pretrained model.")
                
            tokens = self._preprocess_text(text)
            vector = self._sentence_vector(tokens, self.vectorizer)
            prediction = self.classifier.predict([vector])[0]
            
            return {
                0: "not_recommended",
                1: "recommended",
                2: "no_idea"
            }[prediction]
            
        except Exception as e:
            raise Exception(f"Text analysis failed: {str(e)}")
    
    def saveModel(self):
        """save trained model"""
        try:
            if not self.classifier or not self.vectorizer:
                raise Exception("No trained model to save")
                
            joblib.dump(self.classifier, os.path.join(self.model_dir, 'classifier.joblib'))
            self.vectorizer.save(os.path.join(self.model_dir, 'word2vec.model'))
        except IOError as e:
            raise IOError(f"Failed to save model: {str(e)}")
        except Exception as e:
            raise Exception(f"Error while saving model: {str(e)}")
    

    def loadModel(self):
        """reload from package asset or model_dir"""
        try:
            classifier_filename = 'classifier.joblib'
            vectorizer_filename = 'word2vec.model'

            classifier_path = os.path.join(self.model_dir, classifier_filename)
            vectorizer_path = os.path.join(self.model_dir, vectorizer_filename)

            if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
                self.classifier = joblib.load(classifier_path)
                self.vectorizer = Word2Vec.load(vectorizer_path)
                return
            
            try:
                with importlib.resources.path("PerSent.models", classifier_filename) as clf_p, \
                     importlib.resources.path("PerSent.models", vectorizer_filename) as vec_p:
                    self.classifier = joblib.load(clf_p)
                    self.vectorizer = Word2Vec.load(str(vec_p))
            except FileNotFoundError as e:
                raise FileNotFoundError("Model files not found in model_dir *and* package asset")
            except Exception as e:
                raise Exception(f"Failed to load model from package: {str(e)}")
        
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")


        
    def analyzeCSV(self, input_csv, output_path, summary_path=None, text_column=0):
        """
        Analyze sentiment for comments in a CSV file and save results
        
        Parameters:
            input_csv (str): Path to input CSV file
            output_path (str): Path to save output CSV file
            text_column (str/int, optional): Name or index (0-based) of column containing comments. 
                                          Defaults to 0 (first column).
            summary_path (str, optional): Path to save prediction summary report.
                                       If None, no summary will be saved.
                                       
        Returns:
            pd.DataFrame: DataFrame containing analysis results or None if failed
            
        Raises:
            FileNotFoundError: If input file not found
            ValueError: If invalid column specified
            Exception: For other processing errors
        """
        try:
            # Read input CSV
            if not os.path.exists(input_csv):
                raise FileNotFoundError(f"Input file {input_csv} not found")
                
            df = pd.read_csv(input_csv)
            
            # Determine the correct column
            if isinstance(text_column, int):
                # Handle negative indices
                if text_column < 0:
                    text_column = len(df.columns) + text_column
                    
                if text_column >= len(df.columns) or text_column < 0:
                    raise ValueError(f"Column index {text_column} is out of range")
                    
                column_name = df.columns[text_column]
            else:
                if text_column not in df.columns:
                    raise ValueError(f"Column '{text_column}' not found in CSV file")
                column_name = text_column
            
            # Analyze each comment
            tqdm.pandas(desc="Analyzing comments")
            df['sentiment'] = df[column_name].progress_apply(self.analyzeText)
            
            # Save results
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Results saved to {output_path}")
            
            # Generate and save summary if requested
            if summary_path:
                summary = self._generate_summary(df)
                summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
                print(f"Summary report saved to {summary_path}")
            
            return df
            
        except FileNotFoundError as e:
            print(f"Error: File not found - {str(e)}")
            return None
        except ValueError as e:
            print(f"Error: Invalid column specification - {str(e)}")
            return None
        except Exception as e:
            print(f"Error during CSV analysis: {str(e)}")
            return None

    def _generate_summary(self, df):
        """Generate prediction summary statistics"""
        try:
            # Count each sentiment
            counts = df['sentiment'].value_counts().to_dict()
            
            # Create summary dataframe
            summary = pd.DataFrame({
                'Category': [
                    'Recommended',
                    'Not Recommended', 
                    'No Idea',
                    'Total',
                    'Model Accuracy'
                ],
                'Count': [
                    counts.get('recommended', 0),
                    counts.get('not_recommended', 0),
                    counts.get('no_idea', 0),
                    len(df),
                    'N/A'  # Accuracy needs to be calculated during training
                ],
                'Percentage': [
                    f"{100 * counts.get('recommended', 0) / len(df):.2f}%",
                    f"{100 * counts.get('not_recommended', 0) / len(df):.2f}%",
                    f"{100 * counts.get('no_idea', 0) / len(df):.2f}%",
                    '100%',
                    'N/A'
                ]
            })
            
            return summary
        except Exception as e:
            raise Exception(f"Failed to generate summary: {str(e)}")


# Github : RezaGooner
