"""Serialization utilities for JSON and other formats.

This module provides utilities for converting Python objects (especially
numpy/torch types) to JSON-serializable native Python types.
"""

import numpy as np
import torch


def convert_to_serializable(obj):
    """
    Convert numpy/torch types to JSON-serializable Python native types.
    
    Recursively processes nested structures (dicts, lists, tuples) and converts
    numpy/torch types to their Python equivalents:
    - np.float32/64 → float
    - np.int32/64 → int
    - np.ndarray → list
    - np.bool_ → bool
    - torch.Tensor → list
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, tensor, etc.)
        
    Returns:
        JSON-serializable version of the object with same structure
        
    Example:
        >>> metrics = {'roc_auc': np.float32(0.85), 'count': np.int64(100)}
        >>> convert_to_serializable(metrics)
        {'roc_auc': 0.85, 'count': 100}
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj
