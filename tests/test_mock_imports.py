"""Tests for the mock_imports context manager itself."""

import sys
import pytest
from tests.mock_imports import mock_imports


class TestMockImports:

    def test_mocks_injected_during_context(self):
        """Modules should be available in sys.modules inside the with block."""
        with mock_imports("fake_module_abc", "fake_module_xyz"):
            assert "fake_module_abc" in sys.modules
            assert "fake_module_xyz" in sys.modules

    def test_mocks_removed_after_context(self):
        """Mocked modules should not remain in sys.modules after exit."""
        with mock_imports("fake_module_abc"):
            pass
        assert "fake_module_abc" not in sys.modules

    def test_preexisting_module_untouched(self):
        """A module already in sys.modules should not be replaced by a mock."""
        original = sys.modules["os"]
        with mock_imports("os"):
            assert sys.modules["os"] is original
        assert sys.modules["os"] is original

    def test_transitive_imports_cleaned(self):
        """Modules added during the yield (not in the mock list) are also removed."""
        import types
        with mock_imports("fake_parent") as ctx:
            # Simulate a transitive import that happens during the yield
            sys.modules["fake_parent.child"] = types.ModuleType("fake_parent.child")
        assert "fake_parent" not in sys.modules
        assert "fake_parent.child" not in sys.modules
        assert "fake_parent.child" in ctx.mocked_names

    def test_ctx_snapshot_is_frozen(self):
        """The snapshot should reflect sys.modules state before mocking."""
        before = frozenset(sys.modules.keys())
        with mock_imports("fake_module_abc") as ctx:
            assert ctx.snapshot == before

    def test_ctx_mocked_names_populated(self):
        """mocked_names should include both explicit mocks and transitive additions."""
        with mock_imports("fake_module_abc") as ctx:
            pass
        assert "fake_module_abc" in ctx.mocked_names
