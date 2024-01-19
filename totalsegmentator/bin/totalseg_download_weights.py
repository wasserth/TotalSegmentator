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

    parser.add_argument("-t", "--task", choices=["total", "total_fast", "lung_vessels", "cerebral_bleed",
                                                 "hip_implant", "coronary_arteries", "pleural_pericard_effusion",
                                                 "body", "body_fast", "vertebrae_body",
                                                 "heartchambers_highres", "appendicular_bones", "tissue_types", "face"],
                        help="Task for which to download the weights", default="total")

    args = parser.parse_args()

    task_to_id = {
        "total": [291, 292, 293, 294, 295, 298],
        "total_fast": [297, 298],
        "lung_vessels": [258],
        "cerebral_bleed": [150],
        "hip_implant": [260],
        "coronary_arteries": [503],
        "pleural_pericard_effusion": [315],
        "body": [299],
        "body_fast": [300],

        "heartchambers_highres": [301],
        "appendicular_bones": [304],
        "tissue_types": [481],
        "vertebrae_body": [302],
        "face": [303],

        # "liver_vessels": [8]
    }

    setup_totalseg()
    set_config_key("statistics_disclaimer_shown", True)

    for task_id in task_to_id[args.task]:
        print(f"Processing {task_id}...")
        download_pretrained_weights(task_id)


if __name__ == "__main__":
    main()
