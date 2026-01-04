"""
IGWO (Improved Grey Wolf Optimizer) feature selection module.

This module implements the Improved Grey Wolf Optimizer with velocity
and adaptive weights for binary feature selection.
"""

from typing import Tuple, List, Optional, Callable
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging
from tqdm import tqdm
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class IGWO:
    """
    Improved Grey Wolf Optimizer for feature selection.

    Implements GWO with velocity-based updates and adaptive weights
    for selecting optimal feature subsets.

    Algorithm:
        1. Initialize wolf pack (binary feature masks)
        2. Initialize velocities
        3. Evaluate fitness (cross-validation accuracy)
        4. Identify alpha, beta, delta (best 3 wolves)
        5. For each iteration:
           - Update 'a' coefficient
           - For each wolf:
             - Calculate distances and velocities to alpha, beta, delta
             - Calculate adaptive weights based on fitness
             - Update velocity and position
             - Apply sigmoid to binary
             - Evaluate fitness
        6. Return best feature mask

    Attributes:
        n_wolves: Number of wolves in the pack.
        n_iterations: Number of optimization iterations.
        inertia_weight: Velocity inertia coefficient (ω).
        target_features: Target number of features to select.
        cv_folds: Number of cross-validation folds for fitness.
        random_state: Random seed for reproducibility.
        convergence_curve_: Fitness values over iterations.
        best_features_: Binary mask of selected features.
        best_fitness_: Best fitness achieved.
    """

    def __init__(
        self,
        n_wolves: int = 30,
        n_iterations: int = 100,
        inertia_weight: float = 0.9,
        target_features: Optional[int] = None,
        cv_folds: int = 3,
        random_state: int = 42
    ):
        """
        Initialize IGWO optimizer.

        Args:
            n_wolves: Population size.
            n_iterations: Maximum iterations.
            inertia_weight: Velocity inertia (ω).
            target_features: Desired number of features (None for automatic).
            cv_folds: Cross-validation folds for fitness evaluation.
            random_state: Random seed.
        """
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations
        self.inertia_weight = inertia_weight
        self.target_features = target_features
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.convergence_curve_: List[float] = []
        self.best_features_: Optional[np.ndarray] = None
        self.best_fitness_: float = 0.0

        self._rng = np.random.RandomState(random_state)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            x: Input array.

        Returns:
            Sigmoid of input.
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _initialize_population(self, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize wolf population and velocities.

        Args:
            n_features: Number of features.

        Returns:
            Tuple of (population, velocities).
        """
        # Initialize positions (binary feature masks)
        if self.target_features is not None:
            # Initialize with approximately target_features selected
            population = np.zeros((self.n_wolves, n_features), dtype=np.float32)
            for i in range(self.n_wolves):
                # Randomly select target_features
                selected = self._rng.choice(
                    n_features,
                    size=min(self.target_features, n_features),
                    replace=False
                )
                population[i, selected] = 1.0
        else:
            # Random initialization
            population = self._rng.rand(self.n_wolves, n_features).astype(np.float32)
            population = (population > 0.5).astype(np.float32)

        # Initialize velocities to zero
        velocities = np.zeros((self.n_wolves, n_features), dtype=np.float32)

        return population, velocities

    def _evaluate_fitness(
        self,
        wolf: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Evaluate fitness of a wolf (feature subset).

        Args:
            wolf: Binary feature mask.
            X: Feature matrix.
            y: Labels.

        Returns:
            Fitness score (cross-validation accuracy).
        """
        # Get selected features
        selected_features = wolf.astype(bool)
        n_selected = selected_features.sum()

        # Penalize if no features selected or too many features
        if n_selected == 0:
            return 0.0

        # Penalty for large feature sets
        feature_penalty = 0.0
        if self.target_features is not None:
            feature_ratio = n_selected / self.target_features
            if feature_ratio > 1.2:  # Allow 20% deviation
                feature_penalty = 0.1 * (feature_ratio - 1.2)

        # Extract selected features
        X_selected = X[:, selected_features]

        try:
            # Train RandomForest with cross-validation
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=1  # Single job for parallel wolf evaluation
            )

            scores = cross_val_score(
                clf, X_selected, y,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=1
            )

            fitness = scores.mean() - feature_penalty

        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            fitness = 0.0

        return max(0.0, fitness)

    def _evaluate_population(
        self,
        population: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate fitness for entire population.

        Args:
            population: Population of wolves.
            X: Feature matrix.
            y: Labels.

        Returns:
            Array of fitness values.
        """
        fitness_values = np.array([
            self._evaluate_fitness(wolf, X, y)
            for wolf in population
        ])

        return fitness_values

    def _get_leaders(
        self,
        population: np.ndarray,
        fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get alpha, beta, and delta wolves (best 3).

        Args:
            population: Wolf population.
            fitness: Fitness values.

        Returns:
            Tuple of (alpha, beta, delta) wolves.
        """
        sorted_indices = np.argsort(fitness)[::-1]  # Descending order

        alpha = population[sorted_indices[0]].copy()
        beta = population[sorted_indices[1]].copy()
        delta = population[sorted_indices[2]].copy()

        return alpha, beta, delta

    def _calculate_adaptive_weights(
        self,
        fitness_alpha: float,
        fitness_beta: float,
        fitness_delta: float
    ) -> Tuple[float, float, float]:
        """
        Calculate adaptive weights for alpha, beta, delta.

        Args:
            fitness_alpha: Fitness of alpha wolf.
            fitness_beta: Fitness of beta wolf.
            fitness_delta: Fitness of delta wolf.

        Returns:
            Tuple of (w_alpha, w_beta, w_delta).
        """
        total_fitness = fitness_alpha + fitness_beta + fitness_delta

        if total_fitness == 0:
            return 1/3, 1/3, 1/3

        w_alpha = fitness_alpha / total_fitness
        w_beta = fitness_beta / total_fitness
        w_delta = fitness_delta / total_fitness

        return w_alpha, w_beta, w_delta

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> 'IGWO':
        """
        Run IGWO feature selection.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).
            verbose: If True, show progress bar.

        Returns:
            Self for method chaining.
        """
        n_samples, n_features = X.shape
        logger.info(f"Running IGWO on {n_samples} samples with {n_features} features")

        # Initialize population and velocities
        population, velocities = self._initialize_population(n_features)

        # Evaluate initial fitness
        fitness = self._evaluate_population(population, X, y)

        # Get initial leaders
        alpha, beta, delta = self._get_leaders(population, fitness)
        fitness_alpha = fitness[np.argmax(fitness)]
        fitness_beta = sorted(fitness)[-2] if len(fitness) > 1 else fitness_alpha
        fitness_delta = sorted(fitness)[-3] if len(fitness) > 2 else fitness_beta

        # Track convergence
        self.convergence_curve_ = []

        # Main optimization loop
        iterator = range(self.n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="IGWO Iterations")

        for iteration in iterator:
            # Update coefficient a (linearly decreasing from 2 to 0)
            a = 2.0 - iteration * (2.0 / self.n_iterations)

            # Update each wolf
            for i in range(self.n_wolves):
                # Calculate coefficients A and C for each leader
                r1_alpha = self._rng.rand(n_features)
                r2_alpha = self._rng.rand(n_features)
                A_alpha = 2 * a * r1_alpha - a
                C_alpha = 2 * r2_alpha

                r1_beta = self._rng.rand(n_features)
                r2_beta = self._rng.rand(n_features)
                A_beta = 2 * a * r1_beta - a
                C_beta = 2 * r2_beta

                r1_delta = self._rng.rand(n_features)
                r2_delta = self._rng.rand(n_features)
                A_delta = 2 * a * r1_delta - a
                C_delta = 2 * r2_delta

                # Calculate distances to leaders
                D_alpha = np.abs(C_alpha * alpha - population[i])
                D_beta = np.abs(C_beta * beta - population[i])
                D_delta = np.abs(C_delta * delta - population[i])

                # Calculate velocities toward each leader
                V_alpha = -A_alpha * D_alpha
                V_beta = -A_beta * D_beta
                V_delta = -A_delta * D_delta

                # Calculate adaptive weights
                w_alpha, w_beta, w_delta = self._calculate_adaptive_weights(
                    fitness_alpha, fitness_beta, fitness_delta
                )

                # Update velocity with inertia and adaptive weights
                velocities[i] = (
                    self.inertia_weight * velocities[i] +
                    w_alpha * V_alpha +
                    w_beta * V_beta +
                    w_delta * V_delta
                )

                # Update position
                population[i] = population[i] + velocities[i]

                # Apply sigmoid and threshold to binary
                population[i] = (self._sigmoid(population[i]) > 0.5).astype(np.float32)

            # Evaluate new fitness
            fitness = self._evaluate_population(population, X, y)

            # Update leaders
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > fitness_alpha:
                alpha = population[best_idx].copy()
                fitness_alpha = fitness[best_idx]

            # Re-identify all leaders
            alpha, beta, delta = self._get_leaders(population, fitness)
            fitness_alpha = fitness[np.argmax(fitness)]
            sorted_fitness = sorted(fitness, reverse=True)
            fitness_beta = sorted_fitness[1] if len(sorted_fitness) > 1 else fitness_alpha
            fitness_delta = sorted_fitness[2] if len(sorted_fitness) > 2 else fitness_beta

            # Track best fitness
            self.convergence_curve_.append(fitness_alpha)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'Best Fitness': f'{fitness_alpha:.4f}'})

        # Store best solution
        self.best_features_ = alpha.astype(bool)
        self.best_fitness_ = fitness_alpha

        n_selected = self.best_features_.sum()
        logger.info(f"IGWO completed. Selected {n_selected} features with fitness {fitness_alpha:.4f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by selecting features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Transformed feature matrix with selected features.
        """
        if self.best_features_ is None:
            raise ValueError("IGWO must be fitted before transform")

        return X[:, self.best_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit IGWO and transform data.

        Args:
            X: Feature matrix.
            y: Labels.

        Returns:
            Transformed feature matrix.
        """
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> np.ndarray:
        """
        Get indices of selected features.

        Returns:
            Array of selected feature indices.
        """
        if self.best_features_ is None:
            return np.array([], dtype=int)

        return np.where(self.best_features_)[0]

    def save(self, filepath: Path) -> None:
        """
        Save IGWO model to disk.

        Args:
            filepath: Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'best_features_': self.best_features_,
            'best_fitness_': self.best_fitness_,
            'convergence_curve_': self.convergence_curve_,
            'params': {
                'n_wolves': self.n_wolves,
                'n_iterations': self.n_iterations,
                'inertia_weight': self.inertia_weight,
                'target_features': self.target_features,
                'cv_folds': self.cv_folds,
                'random_state': self.random_state,
            }
        }

        joblib.dump(save_data, filepath)
        logger.info(f"Saved IGWO model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'IGWO':
        """
        Load IGWO model from disk.

        Args:
            filepath: Path to load from.

        Returns:
            Loaded IGWO instance.
        """
        save_data = joblib.load(filepath)
        params = save_data['params']

        instance = cls(**params)
        instance.best_features_ = save_data['best_features_']
        instance.best_fitness_ = save_data['best_fitness_']
        instance.convergence_curve_ = save_data['convergence_curve_']

        logger.info(f"Loaded IGWO model from {filepath}")
        return instance
