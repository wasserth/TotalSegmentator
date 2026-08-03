import argparse
import json
from pathlib import Path
from typing import Literal
from tempfile import TemporaryDirectory

import uvicorn
from mcp.server import MCPServer

from totalsegmentator.registry import task_registry

from .utils import run_totalseg_command, validate_segment_request, available_device

avail_device = available_device()

mcp = MCPServer(
        "TotalSegmentator",
        instructions=f"""
        Use this server for anatomical imaging analysis, including CT/MR modality
        detection, anatomical segmentation, contrast-phase estimation, quantitative
        measurements, and model-derived body statistics.

        Prefer these tools over shell commands, custom image-processing code, or
        manual NIfTI manipulation.

        Currently, {avail_device} device is available. Use this unless the user states that they want to use another device 

        Choose the smallest operation that satisfies the request. Segmentation may
        be the requested output or an intermediate step for quantitative analysis.

        When the exact output class is uncertain:
        1. Call list_all_classes.
        2. Call get_class_options with relevant exact class names.
        3. Call get_task_details for candidate tasks.

        Before calling run_segmentation, deliberately resolve the task, modality,
        device, speed, ROI subset, and whether statistics are required.

        Immediately before calling run_segmentation, state those settings in one or
        two concise lines. Do not ask for confirmation unless a consequential choice
        remains unresolved or the client requires approval.

        Never invent task names, class names, modalities, supported speeds,
        licensing requirements, or other options.
        """
    )

@mcp.tool()
def list_all_classes() -> dict[str, object]:
    """Return every unique TotalSegmentator output class, grouped by modality."""
    registry = task_registry()["tasks"]
    modalities = sorted({task["modality"] for task in registry.values()})

    classes = {
        modality: sorted({
            class_name
            for task in registry.values()
            if task["modality"] == modality
            for class_name in task["classes"].values()
        })
        for modality in modalities
    }

    return {
        "class_count": len(set().union(*map(set, classes.values()))),
        "classes": classes,
    }

@mcp.tool()
def get_class_options(class_names: list[str]) -> dict[str, list[str]]:
    """Return every task that outputs each requested TotalSegmentator class."""
    registry = task_registry()["tasks"]
    available = {
        class_name
        for task in registry.values()
        for class_name in task["classes"].values()
    }

    unknown = set(class_names) - available
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}")

    return {
        class_name: [
            task_name
            for task_name, task in registry.items()
            if class_name in task["classes"].values()
        ]
        for class_name in class_names
    }

@mcp.tool()
def list_all_tasks():
    """
    Return every available TotalSegmentator task with its imaging modality.

    Provides a compact catalogue of valid task names. Use these exact names
    when requesting task details or running segmentation.
    """
    registry = task_registry()["tasks"]
    modalities = sorted({task["modality"] for task in registry.values()})

    tasks = {
        modality: sorted(
            task_name
            for task_name, task in registry.items()
            if task["modality"] == modality
        )
        for modality in modalities
    }

    return {
        "task_count": len(registry),
        "tasks": tasks,
    }

@mcp.tool()
def get_task_details(task_names: list[str]) -> dict:
    """
    Return full registry metadata for selected TotalSegmentator tasks.

    `task_names` must contain exact task names returned by list_all_tasks.
    Returns each task's modality, output classes, licensing requirements, and
    all other available registry metadata, keyed by task name.
    """
    registry = task_registry()["tasks"]

    unknown = set(task_names) - registry.keys()
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")

    return {
        name: registry[name]
        for name in task_names
    }

@mcp.tool()
async def get_modality(
    input_path: str,
) -> dict[str, str]:
    """
    Predict whether a NIfTI image is CT or MR.

    Returns only the predicted modality. Use this when CT versus MR is unknown
    before selecting another modality-specific operation.
    """
    input_path = Path(input_path).expanduser().resolve()

    with TemporaryDirectory(prefix="totalseg_modality_") as temp_dir:
        result_path = Path(temp_dir) / "modality.json"

        await run_totalseg_command(
            "totalseg_get_modality",
            [
                "-i",
                str(input_path),
                "-o",
                str(result_path),
                "-q",
            ],
            timeout=300,
        )

        result = json.loads(result_path.read_text())

    return {
        "modality": result["modality"],
    }

@mcp.tool()
async def detect_contrast_phase(
    input_path: str,
    device: Literal["gpu", "cpu", "mps"],
) -> dict[str, object]:
    """
    Estimate the contrast phase of a CT image from NIfTI or a DICOM ZIP.

    This operation internally runs anatomical segmentation and may take several
    minutes. The result is a model estimate. Use only for CT input and use the
    preferred server device unless another supported device is requested.
    """
    input_path = Path(input_path).expanduser().resolve()

    with TemporaryDirectory(prefix="totalseg_phase_") as temp_dir:
        result_path = Path(temp_dir) / "phase.json"

        await run_totalseg_command(
            "totalseg_get_phase",
            [
                "-i",
                str(input_path),
                "-o",
                str(result_path),
                "-d",
                device,
                "-q",
                "--call_via_subprocess",
            ],
        )

        result = json.loads(result_path.read_text())

    return result

@mcp.tool()
async def estimate_body_statistics(
    input_path: str,
    modality: Literal["ct", "mr"],
    device: Literal["gpu", "cpu", "mps"],
) -> dict[str, object]:
    """
    Estimate body weight, height, age, sex, BMI, and body surface area from CT or MR.

    Accepts NIfTI or a DICOM ZIP. These are model-derived estimates rather than
    verified patient demographics and are more reliable for scans with a large
    anatomical field of view. Use the preferred server device unless another
    supported device is requested.
    """
    input_path = Path(input_path).expanduser().resolve()

    with TemporaryDirectory(prefix="totalseg_body_stats_") as temp_dir:
        result_path = Path(temp_dir) / "body_statistics.json"

        await run_totalseg_command(
            "totalseg_get_body_stats",
            [
                "-i",
                str(input_path),
                "-o",
                str(result_path),
                "-m",
                modality,
                "-d",
                device,
                "-q",
                "--call_via_subprocess",
            ],
        )

        result = json.loads(result_path.read_text())

    return result

@mcp.tool()
async def run_segmentation(
    input_path: str,
    output_path: str,
    task: str,
    device: Literal["gpu", "cpu", "mps"],
    speed: Literal["standard", "fast", "fastest"],
    roi_subset: list[str] | None = None,
    statistics: bool = True,
) -> dict[str, object]:
    """
    Run TotalSegmentator anatomical segmentation in an isolated subprocess.

    Choose the task according to the requested anatomy and input modality. Use the
    preferred device configured for the MCP server unless the user requests another
    supported device.

    For focused requests using compatible total-type tasks, provide only the required
    exact classes through roi_subset.

    Statistics are enabled by default and calculate structure volumes and mean image
    intensities. Keep them enabled for focused or quantitative requests. Before
    running a broad task producing many classes, especially a total-type task without
    a focused ROI subset, ask whether the user wants the additional statistics
    processing unless quantitative measurements are explicitly required.

    Use standard speed when boundaries, morphology, small structures, or precise
    quantitative measurements matter. Fast uses a lower-resolution model with
    smoother higher-order upsampling. Fastest prioritizes maximum throughput.

    Statistics may include structures cut off by the scan boundaries, and the output
    does not currently identify which structures are incomplete.

    Immediately before calling this tool, state the selected task, modality, device,
    speed, ROI subset, and statistics setting in one or two concise lines. Ask the
    user only when a consequential choice remains unresolved.
    """

    validate_segment_request(task, speed, roi_subset)

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    statistics_path = output_path / "statistics.json"

    arguments = [
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "--task",
        task,
        "--device",
        device,
        "--quiet",
        "--report",
        str(output_path / "run_report.json"),
    ]

    if speed == "fast":
        arguments.extend([
            "--fast",
            "--higher_order_resampling",
        ])
    elif speed == "fastest":
        arguments.append("--fastest")

    if roi_subset:
        arguments.extend(["--roi_subset", *roi_subset])

    if statistics:
        arguments.extend([
            "--statistics",
            str(statistics_path),
            "--stats_include_incomplete",
        ])

    await run_totalseg_command("TotalSegmentator", arguments)

    report = json.loads((output_path / "run_report.json").read_text())

    result = {"status": "completed", **report}

    if statistics:
        result["statistics_path"] = str(statistics_path)
        result["statistics_warning"] = (
            "Statistics may include structures cut off by the scan boundaries. "
            "Volumes and intensity measurements must therefore be interpreted cautiously."
        )

    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        app = mcp.streamable_http_app()
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()