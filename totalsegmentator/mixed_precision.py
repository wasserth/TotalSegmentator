"""Mixed precision helpers for TotalSegmentator.

We no longer use environment variables. Mixed precision is controlled
explicitly via a runtime flag set by the CLI or API.
"""
import torch

_enabled = False
_status_logged = False


def set_mixed_precision(enabled: bool) -> None:
    """Enable or disable mixed precision for the current process."""
    global _enabled, _status_logged
    _enabled = bool(enabled)
    # allow re-logging on next call when state changes
    _status_logged = False


def want_mixed_precision() -> bool:
    """Return True if mixed precision should be used (flag enabled + CUDA available)."""
    if not torch.cuda.is_available():
        return False
    return _enabled


def log_mixed_precision_status():
    """Log status once. Called after nnUNet env vars are set (python_api)."""
    global _status_logged
    if _status_logged:
        return
    if want_mixed_precision():
        print("TotalSegmentator: mixed precision will be used (autocast).", flush=True)
    else:
        print("TotalSegmentator: mixed precision not requested -> using FP32.", flush=True)
    _status_logged = True

__all__ = ["want_mixed_precision", "log_mixed_precision_status", "set_mixed_precision"]