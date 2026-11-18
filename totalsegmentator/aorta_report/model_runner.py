from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.nifti_ext_header import load_multilabel_nifti


TOTAL_ROIS = [
    "aorta",
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "common_carotid_artery_left",
    "iliac_artery_left",
    "iliac_artery_right",
    "vertebrae_T12",
]

# The input image has the resolution of the model spacing. (T710: 1.5mm, T713: 0.8mm, T716: 0.8mm)
MODEL_RUNS = [
    (
        "Running TotalSegmentator - aorta",
        "  Skipping TotalSeg (already exists)",
        ["brachiocephalic_trunk.nii.gz"],
        ["-ta", "total", "-rs", *TOTAL_ROIS],
    ),
    (
        "Running T716 TF lumen",
        "  Skipping T716 (already exists)",
        ["aorta_true_lumen.nii.gz", "aorta_false_lumen.nii.gz"],
        ["-ta", "aortic_dissection"],
    ),
    (
        "Running T713 annulus proper",
        "  Skipping T713 (already exists)",
        ["annulus_proper.nii.gz", "sinotubular_junction.nii.gz"],
        ["-ta", "aorta_annulus"],
    ),
    (
        "Running T710 renal arteries",
        "  Skipping T710 (already exists)",
        ["celiac_trunk.nii.gz", "superior_mesenteric_artery.nii.gz", "renal_arteries.nii.gz"],
        ["-ta", "renal_arteries"],
    ),
]

MODAL_APP_NAME = "totalsegmentator"
MODAL_SEGMENTATION_FUNCTION = "run_totalsegmentator"
MODAL_NNUNET_FUNCTION = "run_nnunet_v2"
MODAL_PHASE_FUNCTION = "run_get_phase"
MODAL_NNUNET_TASKS = {
    "aortic_dissection": {
        "task_id": 716,
        "model": "3d_fullres_high",
        "tr": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "folds": "0",
    },
    "aorta_annulus": {
        "task_id": 713,
        "model": "3d_fullres_high",
        "tr": "nnUNetTrainer_DASegOrd0",
        "folds": "0",
    },
    "renal_arteries": {
        "task_id": 710,
        "model": "3d_fullres",
        "tr": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "folds": "0",
    },
}


def _command(ct_path: Path, output_dir: Path, task_args: list[str]) -> list[str]:
    return [
        "TotalSegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(output_dir),
        "-ns",
        "1",
        *task_args,
    ]


def _outputs_exist(output_dir: Path, names: list[str]) -> bool:
    return all((output_dir / name).exists() for name in names)


def _check_outputs(output_dir: Path, names: list[str], description: str) -> None:
    missing = [str(output_dir / name) for name in names if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{description} did not create the expected output files:\n  - "
            + "\n  - ".join(missing)
        )


def _pending_model_runs(output_dir: Path, logger):
    """Select incomplete model tasks and report completed cached tasks."""
    pending = []
    for status, skip_message, outputs, task_args in MODEL_RUNS:
        if _outputs_exist(output_dir, outputs):
            logger.info(skip_message)
        else:
            pending.append((status, outputs, task_args))
    return pending


def _check_model_run_outputs(output_dir: Path, model_run) -> None:
    status, outputs, _ = model_run
    _check_outputs(output_dir, outputs, status)


def _run(command: list[str], description: str) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


def _validate_host(host: str) -> None:
    if host not in {"local", "modal"}:
        raise ValueError(f"Unsupported host '{host}'. Expected 'local' or 'modal'.")


def _get_modal_function(function_name):
    try:
        import modal
    except ImportError as exc:
        raise ImportError(
            "Modal execution requires the optional 'modal' package."
        ) from exc
    return modal.Function.from_name(MODAL_APP_NAME, function_name)


def _load_for_remote(ct_path):
    image = nib.load(ct_path)
    return nib.Nifti1Image(
        np.asanyarray(image.dataobj), image.affine, image.header.copy()
    )


def _modal_request(task_args):
    task_idx = task_args.index("-ta")
    task = task_args[task_idx + 1]
    if task in MODAL_NNUNET_TASKS:
        return MODAL_NNUNET_FUNCTION, MODAL_NNUNET_TASKS[task]

    options = {"task": task, "ml": True, "nr_thr_saving": 1}
    if "-rs" in task_args:
        roi_idx = task_args.index("-rs")
        options["roi_subset"] = [
            value for value in task_args[roi_idx + 1 :] if not value.startswith("-")
        ]
    if "--fast" in task_args:
        options["fast"] = True
    return MODAL_SEGMENTATION_FUNCTION, options


def _save_modal_masks(image, output_dir):
    image, label_map = load_multilabel_nifti(image)
    data = np.asanyarray(image.dataobj)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label_id, class_name in label_map.items():
        mask = (data == label_id).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(mask, image.affine, image.header),
            output_dir / f"{class_name}.nii.gz",
        )


def _run_modal(image, function_name, options):
    function = _get_modal_function(function_name)
    return function.remote(image, options)


async def _run_modal_async(image, function_name, options):
    function = _get_modal_function(function_name)
    return await function.remote.aio(image, options)


def get_aorta_fast(ct_path, tmp_dir, logger, totalseg_version="v2", host="local"):
    """Create the fast aorta/heart masks used for cropping."""
    if (tmp_dir / "aorta.nii.gz").exists():
        logger.info("  Skipping TotalSeg aorta fast (already exists)")
        return
    _validate_host(host)
    heart_name = "heart_myocardium" if totalseg_version == "v1" else "heart"
    task_args = ["-ta", "total", "-rs", "aorta", heart_name, "--fast"]
    if host == "local":
        _run(
            _command(Path(ct_path), Path(tmp_dir), task_args),
            "TotalSegmentator aorta fast",
        )
    else:
        function_name, options = _modal_request(task_args)
        image = _run_modal(_load_for_remote(ct_path), function_name, options)
        _save_modal_masks(image, tmp_dir)
    _check_outputs(Path(tmp_dir), ["aorta.nii.gz", f"{heart_name}.nii.gz"], "TotalSegmentator aorta fast")


def run_models_consecutive(ct_path, tmp_dir, logger, host="local"):
    """Run the four registered TotalSegmentator tasks sequentially."""
    _validate_host(host)
    started = time.time()
    ct_path, tmp_dir = Path(ct_path), Path(tmp_dir)
    pending = _pending_model_runs(tmp_dir, logger)
    pending_by_status = {run[0]: run for run in pending}
    remote_image = _load_for_remote(ct_path) if host == "modal" and pending else None
    for status, _, _, _ in MODEL_RUNS:
        yield status
        model_run = pending_by_status.get(status)
        if model_run is None:
            continue
        _, _, task_args = model_run
        if host == "local":
            _run(_command(ct_path, tmp_dir, task_args), status)
        else:
            function_name, options = _modal_request(task_args)
            image = _run_modal(remote_image, function_name, options)
            _save_modal_masks(image, tmp_dir)
        _check_model_run_outputs(tmp_dir, model_run)
    print(f"Models done (took: {time.time() - started:.2f}s)")
    yield "Models done"


async def _run_async(command: list[str], description: str) -> None:
    process = await asyncio.create_subprocess_exec(*command)
    return_code = await process.wait()
    if return_code:
        raise RuntimeError(f"{description} failed with exit code {return_code}")


async def run_models_parallel(ct_path, tmp_dir, logger, host="local"):
    """Run independent registered tasks concurrently, writing per-class masks."""
    _validate_host(host)
    print("Running Models - ASYNC")
    started = time.time()
    ct_path, tmp_dir = Path(ct_path), Path(tmp_dir)
    pending = _pending_model_runs(tmp_dir, logger)
    if pending:
        if host == "local":
            await asyncio.gather(
                *(
                    _run_async(_command(ct_path, tmp_dir, task_args), status)
                    for status, _, task_args in pending
                )
            )
        else:
            remote_image = _load_for_remote(ct_path)
            requests = [_modal_request(task_args) for _, _, task_args in pending]
            images = await asyncio.gather(
                *(
                    _run_modal_async(remote_image, function_name, options)
                    for function_name, options in requests
                )
            )
            for image in images:
                _save_modal_masks(image, tmp_dir)
    for model_run in pending:
        _check_model_run_outputs(tmp_dir, model_run)
    print(f"Models ASYNC done (took: {time.time() - started:.2f}s)")


def get_contrast_phase(ct_path, tmp_dir, logger, host="local"):
    """Run TotalSegmentator's contrast-phase classifier with source-compatible caching."""
    _validate_host(host)
    started = time.time()
    output_path = Path(tmp_dir) / "contrast_phase.json"
    if output_path.exists():
        logger.info("  Skipping totalseg_get_phase (already exists)")
    elif host == "local":
        completed = subprocess.run(
            ["totalseg_get_phase", "-i", str(ct_path), "-o", str(output_path), "-q"],
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"totalseg_get_phase failed with exit code {completed.returncode}"
            )
    else:
        function = _get_modal_function(MODAL_PHASE_FUNCTION)
        result = function.remote(_load_for_remote(ct_path))
        with output_path.open("w") as file:
            json.dump(result, file)
    with output_path.open() as file:
        result = json.load(file)
    print(f"Contrast phase done (took: {time.time() - started:.2f}s)")
    return result["phase"]
