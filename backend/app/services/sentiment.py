"""
Sentiment analysis service using RoBERT and XGBoost models.
"""
import os
import pickle
import joblib
from typing import Dict, Literal
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from app.config import settings


class SentimentAnalyzer:
    """Sentiment analysis using RoBERT and XGBoost models."""

    def __init__(self):
        self.robert_model = None
        self.robert_tokenizer = None
        self.xgboost_model = None
        self.xgboost_preprocessor = None
        self.xgboost_selector = None
        self.svm_model = None
        self.svm_preprocessor = None
        self.svm_vectorizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_robert_model(self):
        """Load RoBERT model."""
        if self.robert_model is None:
            model_path = settings.ROBERT_MODEL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"RoBERT model not found at {model_path}")

            self.robert_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.robert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.robert_model.to(self.device)
            self.robert_model.eval()

    def load_xgboost_model(self):
        """Load XGBoost model."""
        if self.xgboost_model is None:
            model_path = settings.XGBOOST_MODEL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"XGBoost model not found at {model_path}")

            loaded = joblib.load(model_path)

            # Handle case where model is saved as a dictionary
            if isinstance(loaded, dict):
                self.xgboost_model = loaded.get('model')
                # Try to load preprocessor and selector from the dict if available
                if not self.xgboost_preprocessor and 'preprocessor' in loaded:
                    self.xgboost_preprocessor = loaded.get('preprocessor')
                if not self.xgboost_selector and 'selector' in loaded:
                    self.xgboost_selector = loaded.get('selector')
            else:
                self.xgboost_model = loaded

            # Load preprocessor and selector if paths are provided
            if settings.XGBOOST_PREPROCESSOR_PATH and os.path.exists(settings.XGBOOST_PREPROCESSOR_PATH):
                from preprocessing.pipeline import PreprocessingPipeline
                self.xgboost_preprocessor = PreprocessingPipeline.load(settings.XGBOOST_PREPROCESSOR_PATH)

            if settings.XGBOOST_SELECTOR_PATH and os.path.exists(settings.XGBOOST_SELECTOR_PATH):
                from features.selector import FeatureSelector
                self.xgboost_selector = FeatureSelector.load(settings.XGBOOST_SELECTOR_PATH)

    def load_svm_model(self):
        """Load SVM model."""
        if self.svm_model is None:
            model_path = settings.SVM_MODEL_PATH
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"SVM model not found at {model_path}")

            self.svm_model = joblib.load(model_path)

            # Load preprocessor
            if settings.SVM_PREPROCESSOR_PATH and os.path.exists(settings.SVM_PREPROCESSOR_PATH):
                from preprocessing.pipeline import PreprocessingPipeline
                self.svm_preprocessor = PreprocessingPipeline.load(settings.SVM_PREPROCESSOR_PATH)

            # Load vectorizer (TfidfVectorizer for SVM)
            if settings.SVM_VECTORIZER_PATH and os.path.exists(settings.SVM_VECTORIZER_PATH):
                self.svm_vectorizer = joblib.load(settings.SVM_VECTORIZER_PATH)

    def analyze_with_robert(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment using RoBERT model.

        Args:
            text: Review text to analyze

        Returns:
            Dictionary with sentiment label and confidence score
        """
        self.load_robert_model()

        inputs = self.robert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.robert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # Map prediction to label
        # Note: RoBERT Romanian model uses: 0=negative, 1=positive, 2=neutral
        label_map = {0: "negative", 1: "positive", 2: "neutral"}
        sentiment_label = label_map.get(prediction, "unknown")

        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": float(confidence),
            "model_used": "robert"
        }

    def analyze_with_xgboost(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment using XGBoost model.

        Args:
            text: Review text to analyze

        Returns:
            Dictionary with sentiment label and confidence score
        """
        try:
            self.load_xgboost_model()

            # Preprocess text if preprocessor is available
            if self.xgboost_preprocessor:
                text_features = self.xgboost_preprocessor.transform([text])
            else:
                raise Exception("XGBoost preprocessor not available. The model requires a preprocessor to function correctly.")

            # Apply feature selection if selector is available
            if self.xgboost_selector:
                text_features = self.xgboost_selector.transform(text_features)

            # Make prediction
            prediction = self.xgboost_model.predict(text_features)[0]
            probabilities = self.xgboost_model.predict_proba(text_features)[0]
            confidence = float(np.max(probabilities))

            # Map prediction to label (binary classification: 0=negative, 1=positive)
            label_map = {0: "negative", 1: "positive"}
            sentiment_label = label_map.get(int(prediction), "unknown")

            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": confidence,
                "model_used": "xgboost"
            }
        except Exception as e:
            # If XGBoost fails completely, raise a clear error
            raise Exception(f"XGBoost analysis failed: {str(e)}")

    def analyze_with_svm(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment using SVM model.

        Args:
            text: Review text to analyze

        Returns:
            Dictionary with sentiment label and confidence score
        """
        try:
            self.load_svm_model()

            # Preprocess text
            if self.svm_preprocessor:
                text_features = self.svm_preprocessor.transform([text])
            else:
                raise Exception("SVM preprocessor not available. The model requires a preprocessor to function correctly.")

            # Apply TF-IDF vectorization
            if self.svm_vectorizer:
                text_features = self.svm_vectorizer.transform(text_features)
            else:
                raise Exception("SVM vectorizer not available. The model requires a TF-IDF vectorizer to function correctly.")

            # Make prediction
            prediction = self.svm_model.predict(text_features)[0]
            probabilities = self.svm_model.predict_proba(text_features)[0]
            confidence = float(np.max(probabilities))

            # Map prediction to label (binary classification: 0=negative, 1=positive)
            label_map = {0: "negative", 1: "positive"}
            sentiment_label = label_map.get(int(prediction), "unknown")

            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": confidence,
                "model_used": "svm"
            }
        except Exception as e:
            raise Exception(f"SVM analysis failed: {str(e)}")

    def analyze(
        self,
        text: str,
        model: Literal["robert", "xgboost", "svm"] = "robert"
    ) -> Dict[str, any]:
        """
        Analyze sentiment using specified model.

        Args:
            text: Review text to analyze
            model: Model to use ("robert", "xgboost", or "svm")

        Returns:
            Dictionary with sentiment label and confidence score
        """
        if model == "robert":
            return self.analyze_with_robert(text)
        elif model == "xgboost":
            return self.analyze_with_xgboost(text)
        elif model == "svm":
            return self.analyze_with_svm(text)
        else:
            raise ValueError(f"Unknown model: {model}")


# Global instance
sentiment_analyzer = SentimentAnalyzer()
