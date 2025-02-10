#!/usr/bin/env python
import os
import sys
from pathlib import Path
import argparse

from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import setup_totalseg, set_config_key


def main():
    """
    Download totalsegmentator weights

    Info: If want to download models with require a license you have to run `totalseg_set_license` first.
    """
    parser = argparse.ArgumentParser(description="Import manually downloaded weights.",
                                     epilog="Written by Jakob Wasserthal.")

    parser.add_argument("-t", "--task", choices=["total", "total_fast", "total_mr", "total_fast_mr",
                                                 "lung_vessels", "cerebral_bleed",
                                                 "hip_implant", "coronary_arteries", "pleural_pericard_effusion",
                                                 "body", "body_fast", "body_mr", "body_mr_fast", "vertebrae_mr",
                                                 "vertebrae_body",
                                                 "heartchambers_highres", "appendicular_bones", 
                                                 "tissue_types", "tissue_types_mr", "tissue_4_types", "face", "face_mr",
                                                 "head_glands_cavities", "head_muscles", "headneck_bones_vessels",
                                                 "headneck_muscles", "liver_vessels", "brain_structures",
                                                 "lung_nodules", "kidney_cysts", "breasts", "ventricle_parts",
                                                 "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr", 
                                                #  "aortic_sinuses", 
                                                 "all"],
                        help="Task for which to download the weights", default="total")

    args = parser.parse_args()

    task_to_id = {
        "total": [291, 292, 293, 294, 295, 298],
        "total_fast": [297, 298],
        "total_mr": [850, 851],
        "total_fast_mr": [852, 853],
        "lung_vessels": [258],
        "cerebral_bleed": [150],
        "hip_implant": [260],
        "pleural_pericard_effusion": [315],
        "body": [299],
        "body_fast": [300],
        "body_mr": [597],
        "body_mr_fast": [598],
        "vertebrae_mr": [756],
        "head_glands_cavities": [775],
        "headneck_bones_vessels": [776],
        "head_muscles": [777],
        "headneck_muscles": [778, 779],
        "liver_vessels": [8],
        "lung_nodules": [913],
        "kidney_cysts": [789],
        "oculomotor_muscles": [351],
        "breasts": [527],
        "ventricle_parts": [552],

        "heartchambers_highres": [301],
        "appendicular_bones": [304],
        "appendicular_bones_mr": [855],
        "tissue_types": [481],
        "tissue_types_mr": [854],
        "tissue_4_types": [485],
        "vertebrae_body": [305],
        "face": [303],
        "face_mr": [856],
        "brain_structures": [409],
        "thigh_shoulder_muscles": [857],
        "thigh_shoulder_muscles_mr": [857],
        "coronary_arteries": [507],
        # "aortic_sinuses": [920]
    }

    setup_totalseg()
    set_config_key("statistics_disclaimer_shown", True)

    if args.task == "all":
        # Get unique task IDs from all tasks
        all_task_ids = set()
        for task_ids in task_to_id.values():
            if isinstance(task_ids, list):
                all_task_ids.update(task_ids)
            else:
                all_task_ids.add(task_ids)
        
        for task_id in sorted(all_task_ids):
            print(f"Processing {task_id}...")
            download_pretrained_weights(task_id)
    else:
        for task_id in task_to_id[args.task]:
            print(f"Processing {task_id}...")
            download_pretrained_weights(task_id)


if __name__ == "__main__":
    main()
