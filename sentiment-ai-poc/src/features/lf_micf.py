"""
LF-MICF feature extraction module.

This module implements Log-term Frequency - Modified Inverse Class Frequency
for weighted feature extraction.
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import logging
import joblib
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class LF_MICF:
    """
    Log-term Frequency - Modified Inverse Class Frequency feature extractor.

    Implements the LF-MICF weighting scheme which combines log-term frequency
    with a modified inverse class frequency that considers class-specific
    term distributions.

    Mathematical formulation:
        LF-MICF(t) = LTF(t) × MICF(t)
        LTF(t) = log(1 + TF(t))
        MICF(t) = Σ(w_q × log(1 + q/c(t)))
        w_q = log(1 + (r_t↑ × r_t˜) / (max(1, r_t←) × max(1, r_t⌢)))

    Attributes:
        min_df: Minimum document frequency for vocabulary.
        max_df: Maximum document frequency (as fraction) for vocabulary.
        vocabulary_: Mapping from terms to feature indices.
        idf_weights_: MICF weights for each term.
        n_features_: Number of features in vocabulary.
    """

    def __init__(
        self,
        min_df: int = 5,
        max_df: float = 0.8
    ):
        """
        Initialize LF_MICF extractor.

        Args:
            min_df: Minimum document frequency (absolute count).
            max_df: Maximum document frequency (fraction of documents).
        """
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.idf_weights_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def _build_vocabulary(
        self,
        documents: List[str],
        labels: np.ndarray
    ) -> Dict[str, int]:
        """
        Build vocabulary with document frequency filtering.

        Args:
            documents: List of preprocessed text documents.
            labels: Class labels for documents.

        Returns:
            Dictionary mapping terms to indices.
        """
        # Count document frequencies
        doc_freq = Counter()
        for doc in documents:
            unique_terms = set(doc.split())
            doc_freq.update(unique_terms)

        n_docs = len(documents)
        max_df_count = int(self.max_df * n_docs)

        # Filter by min_df and max_df
        vocabulary = {}
        idx = 0
        for term, freq in doc_freq.items():
            if self.min_df <= freq <= max_df_count:
                vocabulary[term] = idx
                idx += 1

        logger.info(f"Built vocabulary with {len(vocabulary)} terms")
        return vocabulary

    def _calculate_micf_weights(
        self,
        documents: List[str],
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Calculate MICF weights for each term in vocabulary.

        Args:
            documents: List of preprocessed text documents.
            labels: Class labels for documents.

        Returns:
            Array of MICF weights for each term.
        """
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        n_terms = len(self.vocabulary_)

        # Initialize containers for class-specific statistics
        # r_t_up[class][term] = number of docs in class with term
        # r_t_down[class][term] = number of docs not in class with term
        # r_t_tilde[class][term] = number of docs in class without term
        # r_t_hat[class][term] = number of docs not in class without term

        class_doc_counts = Counter(labels)
        total_docs = len(documents)

        # Build term-document matrix per class
        term_in_class = defaultdict(lambda: defaultdict(int))
        term_in_other = defaultdict(lambda: defaultdict(int))

        for doc, label in zip(documents, labels):
            terms = set(doc.split())
            for term in terms:
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    term_in_class[label][term_idx] += 1

                    # Count for other classes
                    for other_class in unique_classes:
                        if other_class != label:
                            term_in_other[other_class][term_idx] += 1

        # Calculate MICF for each term
        micf_weights = np.zeros(n_terms)

        for term_idx in range(n_terms):
            micf_sum = 0.0

            for class_label in unique_classes:
                # r_t_up: docs in class with term
                r_t_up = term_in_class[class_label].get(term_idx, 0)

                # r_t_down: docs in other classes with term
                r_t_down = sum(
                    term_in_class[other_class].get(term_idx, 0)
                    for other_class in unique_classes if other_class != class_label
                )

                # r_t_tilde: docs in class without term
                r_t_tilde = class_doc_counts[class_label] - r_t_up

                # r_t_hat: docs in other classes without term
                r_t_hat = (total_docs - class_doc_counts[class_label]) - r_t_down

                # Calculate weight w_q
                numerator = r_t_up * r_t_tilde
                denominator = max(1, r_t_down) * max(1, r_t_hat)
                w_q = np.log(1 + numerator / denominator)

                # Calculate class contribution to MICF
                # c(t) is the number of classes containing term
                c_t = sum(1 for c in unique_classes if term_in_class[c].get(term_idx, 0) > 0)
                c_t = max(1, c_t)

                class_contribution = w_q * np.log(1 + n_classes / c_t)
                micf_sum += class_contribution

            micf_weights[term_idx] = micf_sum

        return micf_weights

    def fit(self, documents: List[str], labels: np.ndarray) -> 'LF_MICF':
        """
        Fit the LF-MICF extractor on training documents.

        Args:
            documents: List of preprocessed text documents.
            labels: Class labels for documents.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting LF-MICF on {len(documents)} documents")

        # Build vocabulary
        self.vocabulary_ = self._build_vocabulary(documents, labels)
        self.n_features_ = len(self.vocabulary_)

        # Calculate MICF weights
        self.idf_weights_ = self._calculate_micf_weights(documents, labels)

        logger.info(f"LF-MICF fitted with {self.n_features_} features")
        return self

    def transform(self, documents: List[str], sparse: bool = True) -> Union[np.ndarray, csr_matrix]:
        """
        Transform documents to LF-MICF weighted feature vectors.

        Args:
            documents: List of preprocessed text documents.
            sparse: If True, return sparse matrix; otherwise dense array.

        Returns:
            Feature matrix of shape (n_documents, n_features).
        """
        if self.vocabulary_ is None or self.idf_weights_ is None:
            raise ValueError("LF_MICF must be fitted before transform")

        n_docs = len(documents)
        n_features = self.n_features_

        # Build term frequency matrix
        if sparse:
            # Use sparse representation for efficiency
            row_indices = []
            col_indices = []
            data = []

            for doc_idx, doc in enumerate(documents):
                # Count term frequencies in document
                term_counts = Counter(doc.split())

                for term, count in term_counts.items():
                    if term in self.vocabulary_:
                        term_idx = self.vocabulary_[term]

                        # Calculate LTF
                        ltf = np.log(1 + count)

                        # Calculate LF-MICF
                        lf_micf = ltf * self.idf_weights_[term_idx]

                        row_indices.append(doc_idx)
                        col_indices.append(term_idx)
                        data.append(lf_micf)

            feature_matrix = csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(n_docs, n_features),
                dtype=np.float32
            )
        else:
            # Dense representation
            feature_matrix = np.zeros((n_docs, n_features), dtype=np.float32)

            for doc_idx, doc in enumerate(documents):
                term_counts = Counter(doc.split())

                for term, count in term_counts.items():
                    if term in self.vocabulary_:
                        term_idx = self.vocabulary_[term]
                        ltf = np.log(1 + count)
                        feature_matrix[doc_idx, term_idx] = ltf * self.idf_weights_[term_idx]

        return feature_matrix

    def fit_transform(
        self,
        documents: List[str],
        labels: np.ndarray,
        sparse: bool = True
    ) -> Union[np.ndarray, csr_matrix]:
        """
        Fit and transform documents in one step.

        Args:
            documents: List of preprocessed text documents.
            labels: Class labels for documents.
            sparse: If True, return sparse matrix; otherwise dense array.

        Returns:
            Feature matrix of shape (n_documents, n_features).
        """
        return self.fit(documents, labels).transform(documents, sparse=sparse)

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names (terms) in vocabulary.

        Returns:
            List of terms ordered by feature index.
        """
        if self.vocabulary_ is None:
            return []

        # Invert vocabulary to get terms by index
        idx_to_term = {idx: term for term, idx in self.vocabulary_.items()}
        return [idx_to_term[i] for i in range(self.n_features_)]

    def get_top_features(self, n: int = 50) -> List[Tuple[str, float]]:
        """
        Get top N features by MICF weight.

        Args:
            n: Number of top features to return.

        Returns:
            List of (term, weight) tuples sorted by weight descending.
        """
        if self.vocabulary_ is None or self.idf_weights_ is None:
            return []

        feature_names = self.get_feature_names()
        feature_weights = [
            (feature_names[i], self.idf_weights_[i])
            for i in range(self.n_features_)
        ]

        # Sort by weight descending
        feature_weights.sort(key=lambda x: x[1], reverse=True)

        return feature_weights[:n]

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save LF-MICF model to disk.

        Args:
            filepath: Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'min_df': self.min_df,
            'max_df': self.max_df,
            'vocabulary_': self.vocabulary_,
            'idf_weights_': self.idf_weights_,
            'n_features_': self.n_features_,
        }

        joblib.dump(save_data, filepath)
        logger.info(f"Saved LF-MICF model to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'LF_MICF':
        """
        Load LF-MICF model from disk.

        Args:
            filepath: Path to load the model from.

        Returns:
            Loaded LF_MICF instance.
        """
        save_data = joblib.load(filepath)

        instance = cls(
            min_df=save_data['min_df'],
            max_df=save_data['max_df']
        )

        instance.vocabulary_ = save_data['vocabulary_']
        instance.idf_weights_ = save_data['idf_weights_']
        instance.n_features_ = save_data['n_features_']

        logger.info(f"Loaded LF-MICF model from {filepath}")
        return instance
