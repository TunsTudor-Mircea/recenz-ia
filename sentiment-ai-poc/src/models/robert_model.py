"""
RoBERT (Romanian BERT) model for sentiment classification.

This module provides a wrapper around Hugging Face's transformers
for fine-tuning RoBERT on Romanian sentiment analysis tasks.
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    # Define placeholders to avoid NameError
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TrainingArguments = None
    Trainer = None
    EarlyStoppingCallback = None
    DataCollatorWithPadding = None
    Dataset = None

logger = logging.getLogger(__name__)


class RoBERTModel:
    """
    Romanian BERT model wrapper for sentiment classification.

    This class provides a high-level interface for training and using
    RoBERT (Romanian BERT) for binary sentiment analysis.

    Attributes:
        model_name: Hugging Face model identifier.
        num_labels: Number of output classes.
        max_length: Maximum sequence length for tokenization.
        device: Torch device (cuda/cpu).
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.
        trainer: Hugging Face Trainer instance.
    """

    def __init__(
        self,
        model_name: str = "dumitrescustefan/bert-base-romanian-cased-v1",
        num_labels: int = 2,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize RoBERT model.

        Args:
            model_name: Hugging Face model identifier.
            num_labels: Number of output classes (2 for binary sentiment).
            max_length: Maximum sequence length for tokenization.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).

        Raises:
            ImportError: If transformers library is not installed.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers accelerate"
            )

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing RoBERT model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )

        # Move model to device
        self.model.to(self.device)

        self.trainer: Optional[Trainer] = None

        logger.info("RoBERT model initialized successfully")

    def _tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize examples for training/evaluation.

        Args:
            examples: Dictionary with 'text' field.

        Returns:
            Tokenized examples.
        """
        return self.tokenizer(
            examples['text'],
            padding=False,  # Use dynamic padding
            truncation=True,
            max_length=self.max_length
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Tuple of (predictions, labels).

        Returns:
            Dictionary of metrics.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None
    ) -> Dataset:
        """
        Prepare dataset for training/evaluation.

        Args:
            texts: List of text strings.
            labels: Optional list of labels (required for training).

        Returns:
            Hugging Face Dataset.
        """
        # Create dataset dictionary
        data_dict = {'text': texts}
        if labels is not None:
            data_dict['label'] = labels

        # Convert to Dataset
        dataset = Dataset.from_dict(data_dict)

        # Tokenize
        dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            desc="Tokenizing"
        )

        return dataset

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
        output_dir: Union[str, Path] = "results/robert_training",
        num_train_epochs: int = 5,
        learning_rate: float = 2e-5,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 32,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        logging_steps: int = 50,
        eval_strategy: str = "epoch",
        save_strategy: str = "epoch",
        save_total_limit: int = 2,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "accuracy",
        early_stopping_patience: int = 2,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Train the RoBERT model.

        Args:
            train_texts: Training texts.
            train_labels: Training labels.
            eval_texts: Evaluation texts (optional).
            eval_labels: Evaluation labels (optional).
            output_dir: Directory to save checkpoints and logs.
            num_train_epochs: Number of training epochs.
            learning_rate: Learning rate.
            per_device_train_batch_size: Training batch size per device.
            per_device_eval_batch_size: Evaluation batch size per device.
            warmup_ratio: Warmup ratio for learning rate scheduler.
            weight_decay: Weight decay for optimizer.
            logging_steps: Log every N steps.
            eval_strategy: Evaluation strategy ('epoch', 'steps', or 'no').
            save_strategy: Save strategy ('epoch', 'steps', or 'no').
            save_total_limit: Maximum number of checkpoints to keep.
            load_best_model_at_end: Load best model at the end of training.
            metric_for_best_model: Metric to use for selecting best model.
            early_stopping_patience: Early stopping patience (epochs).
            seed: Random seed.

        Returns:
            Dictionary with final evaluation metrics.
        """
        logger.info(f"Preparing datasets for training...")
        logger.info(f"Train samples: {len(train_texts)}")

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)

        eval_dataset = None
        if eval_texts is not None and eval_labels is not None:
            logger.info(f"Eval samples: {len(eval_texts)}")
            eval_dataset = self.prepare_dataset(eval_texts, eval_labels)

        # Convert output_dir to Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_dir=str(output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=logging_steps,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=True,
            report_to="none",  # Disable wandb/tensorboard
            fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
            dataloader_num_workers=4,
            seed=seed,
        )

        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Initialize callbacks
        callbacks = []
        if early_stopping_patience > 0 and eval_dataset is not None:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks
        )

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()

        # Log training results
        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")

        # Evaluate on eval set if available
        eval_metrics = {}
        if eval_dataset is not None:
            logger.info("Evaluating on evaluation set...")
            eval_metrics = self.trainer.evaluate(eval_dataset)

            logger.info("\nEvaluation Results:")
            logger.info(f"  Accuracy:  {eval_metrics['eval_accuracy']:.4f}")
            logger.info(f"  Precision: {eval_metrics['eval_precision']:.4f}")
            logger.info(f"  Recall:    {eval_metrics['eval_recall']:.4f}")
            logger.info(f"  F1 Score:  {eval_metrics['eval_f1']:.4f}")

        return eval_metrics

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False
    ) -> Union[int, List[int], Tuple[List[int], np.ndarray]]:
        """
        Predict sentiment for text(s).

        Args:
            texts: Single text or list of texts.
            return_probs: If True, return probabilities along with predictions.

        Returns:
            Predictions (and probabilities if return_probs=True).
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Prepare dataset
        dataset = self.prepare_dataset(texts)

        # Predict using trainer if available, otherwise manual prediction
        if self.trainer is not None:
            predictions = self.trainer.predict(dataset)
            logits = predictions.predictions
        else:
            # Manual prediction
            self.model.eval()
            logits_list = []

            with torch.no_grad():
                for i in range(0, len(texts), 32):  # Batch size 32
                    batch_texts = texts[i:i+32]
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model(**inputs)
                    logits_list.append(outputs.logits.cpu().numpy())

            logits = np.vstack(logits_list)

        # Get predictions
        preds = np.argmax(logits, axis=1).tolist()

        # Get probabilities if requested
        if return_probs:
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
            if single_input:
                return preds[0], probs[0]
            return preds, probs

        if single_input:
            return preds[0]
        return preds

    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Save model and tokenizer.

        Args:
            save_dir: Directory to save model and tokenizer.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_dir}")

        # Save model
        self.model.save_pretrained(save_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)

        logger.info("Model and tokenizer saved successfully")

    @classmethod
    def load(cls, model_dir: Union[str, Path], device: Optional[str] = None) -> 'RoBERTModel':
        """
        Load model from directory.

        Args:
            model_dir: Directory containing saved model.
            device: Device to load model on.

        Returns:
            Loaded RoBERTModel instance.
        """
        model_dir = Path(model_dir)

        logger.info(f"Loading model from {model_dir}")

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)

        # Create instance
        instance = cls.__new__(cls)
        instance.model_name = str(model_dir)
        instance.num_labels = model.config.num_labels
        instance.max_length = 512
        instance.device = device
        instance.tokenizer = tokenizer
        instance.model = model
        instance.trainer = None

        logger.info("Model loaded successfully")

        return instance

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RoBERTModel("
            f"model_name={self.model_name}, "
            f"num_labels={self.num_labels}, "
            f"max_length={self.max_length}, "
            f"device={self.device})"
        )
