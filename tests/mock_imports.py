"""
Context manager for importing modules that have heavy/unavailable dependencies.

Usage:

    from tests.mock_imports import mock_imports

    with mock_imports("torch", "nnunetv2", "totalsegmentator.libs") as ctx:
        from totalsegmentator.nnunet import split_image_into_parts

    # sys.modules is fully restored here.
    # ctx.mocked_names  — set of all module names that were mocked or added
    # ctx.snapshot       — sys.modules keys before mocking

The imported names (split_image_into_parts, etc.) remain usable after the
context manager exits because they're bound in the caller's namespace.
"""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock


@contextmanager
def mock_imports(*module_names):
    """Temporarily inject MagicMock entries into sys.modules for the listed
    module names, yield, then restore sys.modules to its pre-mock state.

    Any modules added to sys.modules during the yield (e.g. transitive
    imports triggered by the caller's ``from X import Y``) are also removed,
    preventing test pollution.

    Yields a SimpleNamespace with:
        .mocked_names  — set of all module names that were mocked or added
        .snapshot       — frozenset of sys.modules keys before mocking
    """
    snapshot = frozenset(sys.modules.keys())
    saved = {}
    for name in module_names:
        saved[name] = sys.modules.get(name)
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    ctx = SimpleNamespace(mocked_names=set(), snapshot=snapshot)
    try:
        yield ctx
    finally:
        # Remove everything added during the yield (transitive imports too)
        added = set(sys.modules.keys()) - snapshot
        for name in added:
            del sys.modules[name]
        # Reinstate anything that existed before mocking
        for name, orig in saved.items():
            if orig is not None:
                sys.modules[name] = orig
        ctx.mocked_names = set(module_names) | added
