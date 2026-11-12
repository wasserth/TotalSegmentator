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
#   837: total 6mm, 836: total 3mm, 872: total_mr 3mm, 307: body 6mm,
#   298: total_v2 6mm, 297: total_v2 3mm, 852: total_v2_mr 3mm
TASK_TO_ID = {
    "total": [831, 832, 833, 834, 835, 837],
    "total_fast": [836, 837],
    "total_v2": [291, 292, 293, 294, 295, 298],
    "total_v2_fast": [297, 298],
    "total_mr": [870, 871, 872],
    "total_fast_mr": [872, 873],
    "total_v2_mr": [850, 851, 852],
    "total_v2_fast_mr": [852, 853],
    "total_highres": [841, 842, 843, 844, 845, 307, 837],
    "lung_vessels": [117, 836],
    "lung_vessels_LEGACY": [258, 837],
    "cerebral_bleed": [150, 837],
    "hip_implant": [260, 837],
    "pleural_pericard_effusion": [315, 837],
    "body": [306],
    "body_fast": [307],
    "body_mr": [622],
    "body_mr_fast": [623],
    "vertebrae_mr": [756],
    "head_glands_cavities": [775, 837],
    "headneck_bones_vessels": [776, 837],
    "head_muscles": [777, 837],
    "headneck_muscles": [778, 779, 837],
    "liver_vessels": [8, 837],
    "lung_nodules": [913, 837],
    "kidney_cysts": [789, 837],
    "oculomotor_muscles": [351, 837],
    "breasts": [527],
    "ventricle_parts": [552, 837],
    "liver_segments": [570, 837],
    "liver_segments_mr": [576, 872],
    "liver_lesions": [591, 836],
    "liver_lesions_mr": [589, 872],
    "craniofacial_structures": [115, 837],
    "abdominal_muscles": [952, 307],
    "teeth": [113, 115, 837],
    "trunk_cavities": [343],
    "brain_aneurysm": [615],

    "heartchambers_highres": [301, 836],
    "appendicular_bones": [838],
    "appendicular_bones_highres": [848],
    "appendicular_bones_mr": [875],
    "tissue_types": [481],
    "tissue_types_mr": [929],
    "tissue_4_types": [485],
    "vertebrae_body": [305],
    "vertebrae_pp": [803],
    "vertebrae_pp_refined": [803, 305],
    "face": [303],
    "face_mr": [856],
    "brain_structures": [409, 837],
    "thigh_shoulder_muscles": [857],
    "thigh_shoulder_muscles_mr": [857],
    "coronary_arteries": [509, 837],
    "coronary_arteries_LEGACY": [507, 837],
    "body_stats_xgb": ["body_stats_xgb"],
    "body_stats_mr": ["body_stats_mr"],
    "body_stats_ct": ["body_stats_ct"],
    "aortic_sinuses": [920, 837],
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
