#!/usr/bin/env python
import argparse

from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import setup_totalseg, set_config_key


# Single source of truth for downloadable tasks. Argparse --task choices are
# derived from this map so the CLI and the id table can never drift apart
# (see issue #589).
#
# Each list holds all weights a run of the task needs, so that the task also works on a
# machine without internet. This includes the model which crops the image to the region of
# interest before the task itself runs (also used for --roi_subset):
#   298: total 6mm, 297: total 3mm, 837: total_v3 6mm, 852: total_mr 3mm, 300: body 6mm
TASK_TO_ID = {
    "total": [291, 292, 293, 294, 295, 298],
    "total_fast": [297, 298],
    "total_v3": [831, 832, 833, 834, 835, 837],
    "total_v3_fast": [836, 837],
    "total_mr": [850, 851, 852],
    "total_fast_mr": [852, 853],
    "lung_vessels": [117, 297],
    "lung_vessels_LEGACY": [258, 298],
    "cerebral_bleed": [150, 298],
    "hip_implant": [260, 298],
    "pleural_pericard_effusion": [315, 298],
    "body": [299],
    "body_fast": [300],
    "body_mr": [597],
    "body_mr_fast": [598],
    "vertebrae_mr": [756],
    "head_glands_cavities": [775, 298],
    "headneck_bones_vessels": [776, 298],
    "head_muscles": [777, 298],
    "headneck_muscles": [778, 779, 298],
    "liver_vessels": [8, 298],
    "lung_nodules": [913, 298],
    "kidney_cysts": [789, 298],
    "oculomotor_muscles": [351, 298],
    "breasts": [527],
    "ventricle_parts": [552, 298],
    "liver_segments": [570, 298],
    "liver_segments_mr": [576, 852],
    "liver_lesions": [591, 297],
    "liver_lesions_mr": [589, 852],
    "craniofacial_structures": [115, 298],
    "abdominal_muscles": [952, 300],
    "teeth": [113, 115, 298],
    "trunk_cavities": [343],
    "brain_aneurysm": [615],

    "heartchambers_highres": [301, 297],
    "appendicular_bones": [304],
    "appendicular_bones_mr": [855],
    "tissue_types": [481],
    "tissue_types_mr": [925],
    "tissue_4_types": [485],
    "vertebrae_body": [305],
    "vertebrae_pp": [803],
    "vertebrae_pp_refined": [803, 305],
    "face": [303],
    "face_mr": [856],
    "brain_structures": [409, 298],
    "thigh_shoulder_muscles": [857],
    "thigh_shoulder_muscles_mr": [857],
    "coronary_arteries": [509, 298],
    "coronary_arteries_LEGACY": [507, 298],
    "body_stats_xgb": ["body_stats_xgb"],
    "body_stats_mr": ["body_stats_mr"],
    "body_stats_ct": ["body_stats_ct"],
    "aortic_sinuses": [920, 298],
    "renal_arteries": [710],
    "aorta_annulus": [713],
    "aortic_dissection": [716],
    "pulmonary_artery_landmarks": [514],
}


def main():
    """
    Download totalsegmentator weights

    Info: If want to download models with require a license you have to run `totalseg_set_license` first.
    """
    parser = argparse.ArgumentParser(description="Import manually downloaded weights.",
                                     epilog="Written by Jakob Wasserthal.")

    parser.add_argument("-t", "--task", choices=list(TASK_TO_ID) + ["all"],
                        help="Task for which to download the weights", default="total")

    args = parser.parse_args()

    setup_totalseg()
    set_config_key("statistics_disclaimer_shown", True)

    if args.task == "all":
        # Get unique task IDs from all tasks
        all_task_ids = set()
        for task_ids in TASK_TO_ID.values():
            if isinstance(task_ids, list):
                all_task_ids.update(task_ids)
            else:
                all_task_ids.add(task_ids)

        for task_id in sorted(all_task_ids, key=str):
            print(f"Processing {task_id}...")
            download_pretrained_weights(task_id)
    else:
        for task_id in TASK_TO_ID[args.task]:
            print(f"Processing {task_id}...")
            download_pretrained_weights(task_id)


if __name__ == "__main__":
    main()
