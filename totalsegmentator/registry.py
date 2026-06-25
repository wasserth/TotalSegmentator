"""
Machine-readable registry of TotalSegmentator's segmentation tasks.

This module is the single source of truth for which segmentation tasks exist,
which anatomical classes each one outputs, whether it requires a license and
whether it operates on CT or MR images.

It intentionally only depends on the pure-data maps in ``map_to_binary`` (no
torch, no model weights), so it can be imported and queried instantly. This
powers the ``totalseg_info`` command and the ``--list-tasks`` / ``--list-classes``
flags of the main CLI, letting humans and automation discover the tool's
capabilities without reading the source code.
"""
import importlib.metadata

from totalsegmentator.map_to_binary import class_map, commercial_models


# Selectable tasks, in the order they are offered on the command line.
# bin/TotalSegmentator.py imports this list for its --task choices so the two
# can never drift apart.
TASKS = [
    "total", "total_v3", "body", "body_mr", "vertebrae_mr",
    "lung_vessels", "lung_vessels_LEGACY", "cerebral_bleed", "hip_implant",
    "coronary_arteries", "coronary_arteries_LEGACY",
    "pleural_pericard_effusion", "test",
    "appendicular_bones", "appendicular_bones_mr", "tissue_types", "heartchambers_highres",
    "face", "vertebrae_body", "vertebrae_pp", "total_mr", "tissue_types_mr", "tissue_4_types", "face_mr",
    "head_glands_cavities", "head_muscles", "headneck_bones_vessels", "headneck_muscles",
    "brain_structures", "liver_vessels", "liver_lesions", "liver_lesions_mr", "oculomotor_muscles",
    "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr", "lung_nodules", "kidney_cysts",
    "breasts", "ventricle_parts", "aortic_sinuses", "liver_segments", "liver_segments_mr",
    "total_highres_test", "craniofacial_structures", "abdominal_muscles", "teeth",
    "trunk_cavities", "brain_aneurysm",
]

# Tasks that operate on MR images but whose name does not end in "_mr".
_MR_TASKS_WITHOUT_SUFFIX = {"brain_aneurysm"}  # TOF-MRI only


def package_version():
    """Installed TotalSegmentator version, or None if not installed (e.g. run from source)."""
    try:
        return importlib.metadata.version("TotalSegmentator")
    except importlib.metadata.PackageNotFoundError:
        return None


def task_modality(task):
    """Return "MR" or "CT" for a task name."""
    if task.endswith("_mr") or task in _MR_TASKS_WITHOUT_SUFFIX:
        return "MR"
    return "CT"


def requires_license(task):
    """Whether the task needs a license (free for non-commercial use)."""
    return task in commercial_models


def get_task_classes(task):
    """Return the {label_index: class_name} map a task outputs.

    Raises KeyError for an unknown task.
    """
    try:
        return dict(class_map[task])
    except KeyError:
        raise KeyError(f"Unknown task: {task!r}. Valid tasks: {', '.join(TASKS)}")


def list_tasks():
    """Summary of every selectable task: name, modality, license flag, #classes."""
    return [
        {
            "name": t,
            "modality": task_modality(t),
            "license_required": requires_license(t),
            "num_classes": len(get_task_classes(t)),
        }
        for t in TASKS
    ]


def task_registry():
    """Full machine-readable capability map for all selectable tasks (JSON-serializable)."""
    return {
        "totalsegmentator_version": package_version(),
        "tasks": {
            t: {
                "modality": task_modality(t),
                "license_required": requires_license(t),
                "classes": {str(idx): name for idx, name in get_task_classes(t).items()},
            }
            for t in TASKS
        },
    }


def format_tasks_table():
    """Human-readable table of all tasks (used by totalseg_info and --list-tasks)."""
    rows = list_tasks()
    name_w = max(len("TASK"), max(len(r["name"]) for r in rows))
    header = f"{'TASK'.ljust(name_w)}  MODALITY  LICENSE  CLASSES"
    lines = [header, "-" * len(header)]
    for r in rows:
        lic = "yes" if r["license_required"] else "no"
        lines.append(f"{r['name'].ljust(name_w)}  {r['modality'].ljust(8)}  {lic.ljust(7)}  {r['num_classes']}")
    lines.append("")
    lines.append(f"{len(rows)} tasks. Licensed tasks need a (free for non-commercial) license: "
                 "https://backend.totalsegmentator.com/license-academic/")
    return "\n".join(lines)


def format_classes_table(task):
    """Human-readable index->name listing of the classes a task outputs."""
    classes = get_task_classes(task)
    lic = "license required" if requires_license(task) else "open license"
    lines = [f"Task '{task}'  [{task_modality(task)}, {lic}, {len(classes)} classes]", ""]
    for idx in sorted(classes):
        lines.append(f"{str(idx).rjust(4)}  {classes[idx]}")
    return "\n".join(lines)
