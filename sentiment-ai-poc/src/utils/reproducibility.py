"""
Reproducibility utilities for ensuring deterministic results.

This module provides functions to set random seeds and configure
libraries for reproducible experiments.
"""

import os
import random
import logging
from typing import Optional

import numpy as np

# TensorFlow is optional (only needed for XGBoost GPU acceleration)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

# PyTorch is used by RoBERT
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, TensorFlow, and PyTorch to ensure
    reproducible results across multiple runs.

    Args:
        seed: Random seed value.

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
    """
    logger.info(f"Setting random seed to {seed}")

    # Set PYTHONHASHSEED environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.debug(f"Set PYTHONHASHSEED={seed}")

    # Python random module
    random.seed(seed)
    logger.debug("Set Python random seed")

    # NumPy random
    np.random.seed(seed)
    logger.debug("Set NumPy random seed")

    # TensorFlow random (if available)
    if TF_AVAILABLE and tf is not None:
        tf.random.set_seed(seed)
        logger.debug("Set TensorFlow random seed")

    # PyTorch random (if available)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.debug("Set PyTorch random seed")

    logger.info("Random seeds set successfully")


def make_deterministic(seed: Optional[int] = 42) -> None:
    """
    Configure environment for deterministic operations.

    Sets random seeds and configures TensorFlow/PyTorch for deterministic operations.
    This may impact performance but ensures reproducibility.

    Args:
        seed: Random seed value. If None, only configures determinism without setting seeds.

    Warning:
        Enabling deterministic operations may reduce performance.

    Example:
        >>> make_deterministic(seed=42)
        >>> # All operations will now be deterministic
    """
    logger.info("Configuring environment for deterministic operations")

    # Set seeds if provided
    if seed is not None:
        set_seed(seed)

    # Configure TensorFlow for deterministic operations (if available)
    if TF_AVAILABLE and tf is not None:
        try:
            # Enable deterministic operations (TensorFlow 2.x)
            tf.config.experimental.enable_op_determinism()
            logger.info("Enabled TensorFlow deterministic operations")
        except AttributeError:
            # Fallback for older TensorFlow versions
            logger.warning(
                "TensorFlow deterministic operations not available. "
                "Consider upgrading to TensorFlow 2.9+ for full determinism."
            )

        # Set additional environment variables for determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        logger.debug("Set TensorFlow environment variables for determinism")

    # Configure PyTorch for deterministic operations (if available)
    if TORCH_AVAILABLE and torch is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Enabled PyTorch deterministic operations")

    logger.info("Deterministic configuration complete")


def configure_gpu(
    memory_growth: bool = True,
    memory_limit: Optional[int] = None,
    device_id: Optional[int] = None
) -> None:
    """
    Configure GPU settings for TensorFlow.

    Args:
        memory_growth: Enable memory growth to prevent TensorFlow from allocating
            all GPU memory at once.
        memory_limit: Limit GPU memory usage to specified MB. If None, no limit is set.
        device_id: GPU device ID to use. If None, uses all available GPUs.

    Example:
        >>> configure_gpu(memory_growth=True, memory_limit=4096)
        >>> # TensorFlow will use at most 4GB of GPU memory
    """
    if not TF_AVAILABLE or tf is None:
        logger.warning("TensorFlow not available. Skipping GPU configuration.")
        return

    logger.info("Configuring GPU settings")

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.warning("No GPUs found. Running on CPU.")
        return

    logger.info(f"Found {len(gpus)} GPU(s)")

    try:
        # Select specific GPU if device_id is provided
        if device_id is not None:
            if device_id >= len(gpus):
                raise ValueError(
                    f"Invalid device_id {device_id}. Only {len(gpus)} GPU(s) available."
                )
            gpus = [gpus[device_id]]
            logger.info(f"Using GPU device {device_id}")

        for gpu in gpus:
            if memory_growth:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Enabled memory growth for {gpu.name}")

            if memory_limit is not None:
                # Set memory limit
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit
                    )]
                )
                logger.info(f"Set memory limit to {memory_limit}MB for {gpu.name}")

    except RuntimeError as e:
        logger.error(f"GPU configuration failed: {e}")
        logger.warning("GPU settings must be configured before TensorFlow initialization")
        raise


def get_system_info() -> dict:
    """
    Get system information for reproducibility tracking.

    Returns:
        Dictionary containing system and library version information.

    Example:
        >>> info = get_system_info()
        >>> print(info.get('tensorflow_version', 'Not installed'))
    """
    info = {
        'python_version': os.sys.version,
        'numpy_version': np.__version__,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not set'),
    }

    if TF_AVAILABLE and tf is not None:
        info['tensorflow_version'] = tf.__version__
        info['tensorflow_gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
        info['tensorflow_gpu_count'] = len(tf.config.list_physical_devices('GPU'))
    else:
        info['tensorflow_version'] = 'Not installed'
        info['tensorflow_gpu_available'] = False
        info['tensorflow_gpu_count'] = 0

    if TORCH_AVAILABLE and torch is not None:
        info['torch_version'] = torch.__version__
        info['torch_cuda_available'] = torch.cuda.is_available()
        info['torch_cuda_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        info['torch_version'] = 'Not installed'
        info['torch_cuda_available'] = False
        info['torch_cuda_count'] = 0

    logger.debug(f"System info: {info}")
    return info


def log_system_info() -> None:
    """
    Log system information for reproducibility.

    Example:
        >>> log_system_info()
        # Logs system and library version information
    """
    info = get_system_info()

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Python version: {info['python_version']}")
    logger.info(f"NumPy version: {info['numpy_version']}")
    logger.info(f"TensorFlow version: {info['tensorflow_version']}")
    logger.info(f"GPU available: {info['tensorflow_gpu_available']}")
    logger.info(f"GPU count: {info['tensorflow_gpu_count']}")
    logger.info(f"PYTHONHASHSEED: {info['pythonhashseed']}")
    logger.info("=" * 60)


def setup_reproducible_environment(
    seed: int = 42,
    use_gpu: bool = True,
    memory_growth: bool = True,
    memory_limit: Optional[int] = None,
    device_id: Optional[int] = None,
    log_info: bool = True
) -> None:
    """
    Set up a fully reproducible environment.

    Convenience function that configures seeds, determinism, and GPU settings.

    Args:
        seed: Random seed value.
        use_gpu: Whether to configure GPU. If False, forces CPU usage.
        memory_growth: Enable GPU memory growth.
        memory_limit: GPU memory limit in MB.
        device_id: GPU device ID to use.
        log_info: Whether to log system information.

    Example:
        >>> setup_reproducible_environment(seed=42, memory_limit=4096)
        >>> # Environment is now fully configured for reproducible experiments
    """
    logger.info("Setting up reproducible environment")

    # Log system info first
    if log_info:
        log_system_info()

    # Set up deterministic behavior
    make_deterministic(seed=seed)

    # Configure GPU or force CPU
    if use_gpu:
        configure_gpu(
            memory_growth=memory_growth,
            memory_limit=memory_limit,
            device_id=device_id
        )
    else:
        # Force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("Forcing CPU usage (GPU disabled)")

    logger.info("Reproducible environment setup complete")


def reset_random_state(seed: int = 42) -> None:
    """
    Reset random state during execution.

    Useful when you need to reset randomness in the middle of an experiment.

    Args:
        seed: Random seed value.

    Example:
        >>> reset_random_state(42)
        >>> # Random state has been reset
    """
    logger.info(f"Resetting random state with seed {seed}")
    set_seed(seed)
