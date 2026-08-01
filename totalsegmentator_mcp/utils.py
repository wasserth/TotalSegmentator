import asyncio
import os
import signal
from shutil import which

from totalsegmentator.map_tasks_config import TASK_CONFIGS
from totalsegmentator.registry import task_registry

def available_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "gpu"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop TotalSegmentator and any worker processes it created as gracefully as possible"""
    
    # see if process already killed
    if process.returncode is not None:
        return

    # try to kill politely
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM) # since process is with separate pid, SIGTERM sent to that process
        else:
            process.terminate()
    except ProcessLookupError:
        await process.wait()
        return

    # wait for 10 seconds
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
        return
    except asyncio.TimeoutError:
        pass

    # try to kill forcefully
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        await process.wait()
        return

    await process.wait()

async def run_totalseg_command(
    executable_name: str,
    arguments: list[str],
    timeout: int = 3600,
) -> tuple[str, str]:
    """Run a TotalSegmentator CLI command in an isolated subprocess."""
    
    executable = which(executable_name)
    if executable is None:
        raise RuntimeError(f"{executable_name!r} was not found in PATH.")

    # Put TotalSegmentator and its spawned workers in their own process group separately from the MCP server
    process_options = {}
    if os.name == "posix":
        process_options["start_new_session"] = True

    process = await asyncio.create_subprocess_exec(
        executable,
        *arguments,
        stdin=asyncio.subprocess.DEVNULL, # no keyboard input
        stdout=asyncio.subprocess.PIPE, # capture normal prints in stdout
        stderr=asyncio.subprocess.PIPE, # capture errors in stderr
        **process_options,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )

    except asyncio.TimeoutError as error:
        await stop_process(process)
        raise RuntimeError(
            f"{executable_name} exceeded the execution timeout."
        ) from error

    except asyncio.CancelledError:
        await stop_process(process)
        raise

    stdout = stdout_bytes.decode(errors="replace").strip()
    stderr = stderr_bytes.decode(errors="replace").strip()

    if process.returncode != 0:
        error_message = stderr or stdout or "No error output was produced."

        raise RuntimeError(
            f"{executable_name} failed with exit code "
            f"{process.returncode}:\n{error_message[-4000:]}"
        )

    return stdout, stderr

def supported_speeds(task: str) -> set[str]:
    """Return the inference speeds explicitly implemented for a task."""
    modes = set(TASK_CONFIGS[task].get("sub_modes", {}))
    return {"standard"} | (modes & {"fast", "fastest"})


def validate_segment_request(
    task: str,
    speed: str,
    roi_subset: list[str] | None,
) -> None:
    """Validate task, speed, and ROI compatibility."""
    registry = task_registry()["tasks"]
    if task not in registry:
        raise ValueError(f"Unknown task: {task!r}")

    unknown_rois = set(roi_subset or []) - set(registry[task]["classes"].values())
    if unknown_rois:
        raise ValueError(f"Task {task!r} does not output these classes: {sorted(unknown_rois)}")
    if roi_subset and not task.startswith("total"):
        raise ValueError("roi_subset is supported only for tasks beginning with 'total'.")

    speeds = supported_speeds(task)
    if speed not in speeds:
        raise ValueError(f"Task {task!r} supports these speeds: {sorted(speeds)}")