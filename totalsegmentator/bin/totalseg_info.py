#!/usr/bin/env python
import sys
import json
import argparse

from totalsegmentator.registry import (
    TASKS, list_tasks, get_task_classes, task_registry,
    format_tasks_table, format_classes_table,
)


def main():
    """
    Print machine- and human-readable information about TotalSegmentator's tasks.

    Lists the available segmentation tasks, the anatomical classes each task
    outputs, the modality (CT/MR) and whether a license is required. Runs
    instantly: it needs no GPU and downloads no model weights, which makes it a
    convenient way for scripts and AI coding agents to discover what the tool can
    do (e.g. valid --roi_subset class names) without reading the source code.

    Usage:
    totalseg_info --list-tasks
    totalseg_info --classes -ta total
    totalseg_info --json                 # full capability registry as JSON
    totalseg_info --classes -ta total --json
    """
    parser = argparse.ArgumentParser(
        description="Show information about TotalSegmentator's segmentation tasks and classes.",
        epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-lt", "--list-tasks", action="store_true",
                       help="List all available tasks with modality, license and number of classes.")
    group.add_argument("-cl", "--classes", action="store_true",
                       help="List the classes (index -> name) of the task given via --task.")

    parser.add_argument("-ta", "--task", choices=TASKS, metavar="task",
                        help="Task to show the classes for (used with --classes).")

    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Emit output as JSON instead of a human-readable table.")

    args = parser.parse_args()

    # --classes requires a task
    if args.classes and args.task is None:
        parser.error("--classes requires --task/-ta (e.g. totalseg_info --classes -ta total)")

    if args.classes:
        if args.as_json:
            classes = get_task_classes(args.task)
            print(json.dumps({str(idx): name for idx, name in classes.items()}, indent=2))
        else:
            print(format_classes_table(args.task))
        return

    if args.list_tasks:
        if args.as_json:
            print(json.dumps(list_tasks(), indent=2))
        else:
            print(format_tasks_table())
        return

    # No explicit mode selected.
    if args.as_json:
        # Default JSON output is the full capability registry: one call gives an
        # agent the complete picture (tasks, modalities, license flags, classes).
        print(json.dumps(task_registry(), indent=2))
        return

    # Default human output: the task overview, plus a hint for the other modes.
    print(format_tasks_table())
    print()
    print("Use '--classes -ta <task>' to list a task's classes, or '--json' for machine-readable output.")


if __name__ == "__main__":
    main()
