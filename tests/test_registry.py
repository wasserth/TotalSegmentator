import sys
import json
import subprocess
import unittest

import pytest

from totalsegmentator.registry import (
    TASKS, task_modality, requires_license, get_task_classes,
    list_tasks, task_registry, format_tasks_table, format_classes_table,
)
from totalsegmentator.map_to_binary import commercial_models


class TestRegistry(unittest.TestCase):
    """Pure-data tests for the task registry. No torch / GPU / model weights needed."""

    def test_tasks_unique(self):
        self.assertEqual(len(TASKS), len(set(TASKS)), "TASKS contains duplicates")

    def test_every_task_has_classes(self):
        for t in TASKS:
            classes = get_task_classes(t)
            self.assertIsInstance(classes, dict)
            self.assertGreater(len(classes), 0, f"task '{t}' has no classes")

    def test_known_examples(self):
        self.assertEqual(task_modality("total"), "CT")
        self.assertFalse(requires_license("total"))
        self.assertEqual(len(get_task_classes("total")), 117)
        self.assertEqual(len(get_task_classes("total_v3")), 117)
        self.assertEqual(get_task_classes("total_v3")[26], "vertebrae_L6")
        self.assertNotIn("vertebrae_S1", get_task_classes("total_v3").values())
        self.assertEqual(task_modality("total_mr"), "MR")
        self.assertEqual(task_modality("body_mr"), "MR")
        # TOF-MRI task whose name does not end in "_mr"
        self.assertEqual(task_modality("brain_aneurysm"), "MR")
        self.assertTrue(requires_license("tissue_types"))
        self.assertTrue(requires_license("vertebrae_pp"))
        self.assertEqual(len(get_task_classes("vertebrae_pp")), 24)
        self.assertEqual(get_task_classes("vertebrae_pp")[1], "vertebrae_C1")
        self.assertEqual(get_task_classes("vertebrae_pp")[24], "vertebrae_L5")
        self.assertTrue(requires_license("vertebrae_pp_refined"))
        self.assertEqual(get_task_classes("vertebrae_pp_refined"), get_task_classes("vertebrae_pp"))

    def test_license_flag_matches_commercial_models(self):
        # Every licensed task is listed, and every commercial model is a selectable task.
        self.assertTrue(set(commercial_models).issubset(set(TASKS)))
        for t in TASKS:
            self.assertEqual(requires_license(t), t in commercial_models)

    def test_legacy_task_has_own_classes(self):
        # LEGACY variants are real tasks with their own (older) class map.
        legacy = get_task_classes("lung_vessels_LEGACY")
        self.assertGreater(len(legacy), 0)
        self.assertNotEqual(legacy, get_task_classes("lung_vessels"))

    def test_unknown_task_raises(self):
        with self.assertRaises(KeyError):
            get_task_classes("does_not_exist")

    def test_list_tasks_shape(self):
        rows = list_tasks()
        self.assertEqual(len(rows), len(TASKS))
        for r in rows:
            self.assertEqual(set(r), {"name", "modality", "license_required", "num_classes"})
            self.assertIn(r["modality"], {"CT", "MR"})

    def test_task_registry_json_roundtrip(self):
        reg = task_registry()
        dumped = json.dumps(reg)  # must be JSON-serializable
        reloaded = json.loads(dumped)
        self.assertEqual(set(reloaded["tasks"]), set(TASKS))
        total = reloaded["tasks"]["total"]
        self.assertEqual(total["modality"], "CT")
        self.assertEqual(total["classes"]["1"], "spleen")

    def test_format_helpers_run(self):
        self.assertIn("total", format_tasks_table())
        self.assertIn("spleen", format_classes_table("total"))


class TestTotalsegInfoCLI(unittest.TestCase):
    """Smoke tests for the totalseg_info command (no heavy dependencies)."""

    def _run(self, *args):
        return subprocess.run([sys.executable, "-m", "totalsegmentator.bin.totalseg_info", *args],
                              capture_output=True, text=True)

    def test_list_tasks(self):
        r = self._run("--list-tasks")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("total", r.stdout)

    def test_json_registry(self):
        r = self._run("--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(len(data["tasks"]), len(TASKS))

    def test_classes_for_task(self):
        r = self._run("--classes", "-ta", "total", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data["1"], "spleen")

    def test_classes_without_task_errors(self):
        r = self._run("--classes")
        self.assertEqual(r.returncode, 2)


class TestRunReport(unittest.TestCase):
    """Tests for the run-report builder. Needs torch + nibabel (present in CI)."""

    def test_build_run_report(self):
        pytest.importorskip("torch")
        pytest.importorskip("nibabel")
        from totalsegmentator.python_api import build_run_report

        report = build_run_report(
            input="ct.nii.gz", output=None, task="total", device="cpu",
            fast=False, fastest=False, ml=False, output_type="nifti",
            roi_subset=["liver", "spleen"], runtime_seconds=12.345,
            save_lowres=False)

        expected_keys = {
            "totalsegmentator_version", "nnunetv2_version", "torch_version",
            "task", "modality", "license_required", "device", "fast", "fastest",
            "save_lowres", "multilabel", "output_type", "roi_subset", "input", "output",
            "num_classes", "classes", "runtime_seconds", "output_files",
        }
        self.assertEqual(set(report), expected_keys)
        self.assertEqual(report["task"], "total")
        self.assertEqual(report["modality"], "CT")
        self.assertFalse(report["license_required"])
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["input"], "ct.nii.gz")
        self.assertIsNone(report["output"])
        self.assertEqual(report["runtime_seconds"], 12.35)
        # roi_subset filters the class list down to the requested names.
        self.assertEqual(report["num_classes"], 2)
        self.assertEqual(set(report["classes"].values()), {"liver", "spleen"})
        # report is JSON-serializable
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
