"""
Configuration management utilities.

This module provides functions to load, merge, validate, and export
configuration files for experiments.
"""

import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import copy
import json

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        >>> config = load_config('config/base_config.yaml')
        >>> print(config['model']['lstm_units_1'])
        128
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            logger.warning(f"Config file is empty: {config_path}")
            return {}

        logger.debug(f"Loaded config keys: {list(config.keys())}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config {config_path}: {e}")
        raise


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    The override_config takes precedence over base_config. Performs deep merge
    for nested dictionaries.

    Args:
        base_config: Base configuration dictionary.
        override_config: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.

    Example:
        >>> base = {'model': {'units': 128, 'dropout': 0.5}, 'batch_size': 32}
        >>> override = {'model': {'units': 256}}
        >>> merged = merge_configs(base, override)
        >>> print(merged['model']['units'])  # 256
        >>> print(merged['model']['dropout'])  # 0.5
    """
    # Deep copy to avoid modifying original configs
    merged = copy.deepcopy(base_config)

    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                base[key] = _deep_merge(base[key], value)
            else:
                # Override the value
                base[key] = copy.deepcopy(value)
        return base

    merged = _deep_merge(merged, override_config)
    logger.debug(f"Merged configs with {len(override_config)} override keys")

    return merged


def load_and_merge_configs(
    config_paths: List[Union[str, Path]]
) -> Dict[str, Any]:
    """
    Load and merge multiple configuration files.

    Configurations are merged in order, with later configs overriding earlier ones.

    Args:
        config_paths: List of paths to configuration files.

    Returns:
        Merged configuration dictionary.

    Example:
        >>> config = load_and_merge_configs([
        ...     'config/base.yaml',
        ...     'config/experiment_1.yaml'
        ... ])
    """
    if not config_paths:
        raise ValueError("At least one config path must be provided")

    logger.info(f"Loading and merging {len(config_paths)} config files")

    # Load first config as base
    merged_config = load_config(config_paths[0])

    # Merge remaining configs
    for config_path in config_paths[1:]:
        override_config = load_config(config_path)
        merged_config = merge_configs(merged_config, override_config)

    logger.info("Successfully merged all configurations")
    return merged_config


def validate_config(
    config: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    validation_rules: Optional[Dict[str, callable]] = None
) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required keys (supports nested keys with dot notation).
        validation_rules: Dictionary mapping keys to validation functions.
            Each function should take a value and return True if valid.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If validation fails.

    Example:
        >>> config = {'model': {'units': 128}, 'learning_rate': 0.001}
        >>> validate_config(
        ...     config,
        ...     required_keys=['model.units', 'learning_rate'],
        ...     validation_rules={'learning_rate': lambda x: 0 < x < 1}
        ... )
        True
    """
    def _get_nested_value(d: Dict[str, Any], key_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key_path.split('.')
        value = d
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise KeyError(f"Key path not found: {key_path}")
            value = value[key]
        return value

    # Check required keys
    if required_keys:
        logger.debug(f"Validating {len(required_keys)} required keys")
        for key in required_keys:
            try:
                _get_nested_value(config, key)
            except KeyError:
                error_msg = f"Required config key missing: {key}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    # Apply validation rules
    if validation_rules:
        logger.debug(f"Applying {len(validation_rules)} validation rules")
        for key, validation_func in validation_rules.items():
            try:
                value = _get_nested_value(config, key)
                if not validation_func(value):
                    error_msg = f"Validation failed for key '{key}' with value '{value}'"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            except KeyError:
                # Skip validation if key doesn't exist (will be caught by required_keys check)
                continue

    logger.info("Configuration validation passed")
    return True


def export_config(
    config: Dict[str, Any],
    export_dir: Union[str, Path],
    filename: str = 'config.yaml',
    export_formats: Optional[List[str]] = None
) -> List[Path]:
    """
    Export configuration to experiment directory.

    Args:
        config: Configuration dictionary to export.
        export_dir: Directory to export the config to.
        filename: Base filename for the config (without extension for multiple formats).
        export_formats: List of formats to export ('yaml', 'json').
            Defaults to ['yaml'].

    Returns:
        List of paths to exported config files.

    Example:
        >>> config = {'model': {'units': 128}, 'batch_size': 32}
        >>> paths = export_config(
        ...     config,
        ...     'experiments/exp_001',
        ...     'config',
        ...     export_formats=['yaml', 'json']
        ... )
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if export_formats is None:
        export_formats = ['yaml']

    exported_paths = []

    # Remove extension from filename if present
    base_filename = Path(filename).stem

    for fmt in export_formats:
        if fmt == 'yaml':
            export_path = export_dir / f"{base_filename}.yaml"
            with open(export_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2
                )
            logger.info(f"Exported config to YAML: {export_path}")
            exported_paths.append(export_path)

        elif fmt == 'json':
            export_path = export_dir / f"{base_filename}.json"
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Exported config to JSON: {export_path}")
            exported_paths.append(export_path)

        else:
            logger.warning(f"Unsupported export format: {fmt}")

    return exported_paths


def get_nested_config(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get a value from nested config using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the value (e.g., 'model.lstm.units').
        default: Default value if key path not found.

    Returns:
        The value at the key path, or default if not found.

    Example:
        >>> config = {'model': {'lstm': {'units': 128}}}
        >>> get_nested_config(config, 'model.lstm.units')
        128
        >>> get_nested_config(config, 'model.gru.units', default=64)
        64
    """
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_nested_config(
    config: Dict[str, Any],
    key_path: str,
    value: Any
) -> Dict[str, Any]:
    """
    Set a value in nested config using dot notation.

    Args:
        config: Configuration dictionary to modify.
        key_path: Dot-separated path to the value (e.g., 'model.lstm.units').
        value: Value to set.

    Returns:
        Modified configuration dictionary.

    Example:
        >>> config = {'model': {'lstm': {'units': 128}}}
        >>> set_nested_config(config, 'model.lstm.units', 256)
        >>> config['model']['lstm']['units']
        256
    """
    keys = key_path.split('.')
    current = config

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise ValueError(f"Cannot set nested value: '{key}' is not a dictionary")
        current = current[key]

    # Set the final value
    current[keys[-1]] = value

    return config
