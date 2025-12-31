"""
Logging utilities for the sentiment analysis project.

This module provides functions to set up logging with file and console handlers,
and decorators for timing function execution.
"""

import logging
import time
import functools
from typing import Optional, Callable, Any
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    log_filename: Optional[str] = None,
    propagate: bool = False
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_dir: Directory to save log files. If None, only console logging is used.
        level: Default logging level for both handlers.
        console_level: Logging level for console handler. If None, uses level.
        file_level: Logging level for file handler. If None, uses level.
        log_filename: Name of the log file. If None, generates from logger name and timestamp.
        propagate: Whether to propagate logs to parent loggers.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger('my_module', log_dir='logs', level=logging.DEBUG)
        >>> logger.info('This is an info message')
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    logger.propagate = propagate

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set handler levels
    if console_level is None:
        console_level = level
    if file_level is None:
        file_level = level

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir is provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename if not provided
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = name.replace('.', '_')
            log_filename = f"{safe_name}_{timestamp}.log"

        log_path = log_dir / log_filename

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        logger.debug(f"Logger '{name}' initialized with log file: {log_path}")
    else:
        logger.debug(f"Logger '{name}' initialized (console only)")

    return logger


def get_experiment_logger(
    experiment_name: str,
    experiment_dir: Path,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Create a logger for a specific experiment.

    Args:
        experiment_name: Name of the experiment.
        experiment_dir: Directory for the experiment (logs will be saved here).
        level: Logging level.

    Returns:
        Configured logger for the experiment.

    Example:
        >>> logger = get_experiment_logger('exp_001', Path('experiments/exp_001'))
        >>> logger.info('Starting experiment')
    """
    log_dir = experiment_dir / 'logs'
    log_filename = f"{experiment_name}.log"

    logger = setup_logger(
        name=f"experiment.{experiment_name}",
        log_dir=log_dir,
        level=level,
        log_filename=log_filename
    )

    return logger


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Logs the execution time of the decorated function.

    Args:
        func: Function to time.

    Returns:
        Wrapped function.

    Example:
        >>> @timer
        ... def process_data(data):
        ...     # Process data
        ...     pass
        >>> process_data(my_data)
        # Logs: Function 'process_data' took 2.34 seconds
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()

        logger.debug(f"Starting function '{func.__name__}'")

        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            logger.info(
                f"Function '{func.__name__}' completed in {elapsed_time:.2f} seconds"
            )

            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"Function '{func.__name__}' failed after {elapsed_time:.2f} seconds: {e}"
            )
            raise

    return wrapper


def log_execution(
    log_args: bool = False,
    log_result: bool = False,
    level: int = logging.INFO
) -> Callable:
    """
    Decorator to log function execution with optional argument and result logging.

    Args:
        log_args: Whether to log function arguments.
        log_result: Whether to log function return value.
        level: Logging level for the execution log.

    Returns:
        Decorator function.

    Example:
        >>> @log_execution(log_args=True, log_result=True)
        ... def add(a, b):
        ...     return a + b
        >>> result = add(2, 3)
        # Logs: Calling 'add' with args=(2, 3), kwargs={}
        # Logs: Function 'add' returned 5
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)

            # Log function call with arguments
            if log_args:
                logger.log(
                    level,
                    f"Calling '{func.__name__}' with args={args}, kwargs={kwargs}"
                )
            else:
                logger.log(level, f"Calling '{func.__name__}'")

            # Execute function
            result = func(*args, **kwargs)

            # Log result
            if log_result:
                logger.log(level, f"Function '{func.__name__}' returned {result}")

            return result

        return wrapper
    return decorator


class TimingContext:
    """
    Context manager for timing code blocks.

    Example:
        >>> with TimingContext('data processing'):
        ...     # Process data
        ...     pass
        # Logs: Operation 'data processing' took 1.23 seconds
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO
    ):
        """
        Initialize timing context.

        Args:
            operation_name: Name of the operation being timed.
            logger: Logger to use. If None, uses root logger.
            level: Logging level for the timing message.
        """
        self.operation_name = operation_name
        self.logger = logger if logger is not None else logging.getLogger()
        self.level = level
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.debug(f"Starting operation '{self.operation_name}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        elapsed_time = time.time() - self.start_time

        if exc_type is not None:
            self.logger.error(
                f"Operation '{self.operation_name}' failed after "
                f"{elapsed_time:.2f} seconds"
            )
        else:
            self.logger.log(
                self.level,
                f"Operation '{self.operation_name}' took {elapsed_time:.2f} seconds"
            )

        return False  # Don't suppress exceptions


def set_log_level(logger_name: str, level: int) -> None:
    """
    Set the logging level for a specific logger.

    Args:
        logger_name: Name of the logger to configure.
        level: Logging level to set.

    Example:
        >>> set_log_level('my_module', logging.DEBUG)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


def add_file_handler(
    logger_name: str,
    log_file: Path,
    level: int = logging.INFO
) -> None:
    """
    Add a file handler to an existing logger.

    Args:
        logger_name: Name of the logger.
        log_file: Path to the log file.
        level: Logging level for the file handler.

    Example:
        >>> add_file_handler('my_module', Path('logs/additional.log'))
    """
    logger = logging.getLogger(logger_name)

    # Create file handler
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.debug(f"Added file handler: {log_file}")
