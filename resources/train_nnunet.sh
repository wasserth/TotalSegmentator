#!/bin/bash
set -e

python convert_dataset_to_nnunet.py \
/mnt/nor/wasserthalj_data/TotalSegmentator/zenodo_upload/Totalsegmentator_dataset \
/mnt/nor/nnunet/raw_v2/Dataset101_TotalSegmentator_public_part1 \
class_map_part_organs

python convert_dataset_to_nnunet.py \
/mnt/nor/wasserthalj_data/TotalSegmentator/zenodo_upload/Totalsegmentator_dataset \
/mnt/nor/nnunet/raw_v2/Dataset102_TotalSegmentator_public_part2 \
class_map_part_vertebrae

python convert_dataset_to_nnunet.py \
/mnt/nor/wasserthalj_data/TotalSegmentator/zenodo_upload/Totalsegmentator_dataset \
/mnt/nor/nnunet/raw_v2/Dataset103_TotalSegmentator_public_part3 \
class_map_part_cardiac

python convert_dataset_to_nnunet.py \
/mnt/nor/wasserthalj_data/TotalSegmentator/zenodo_upload/Totalsegmentator_dataset \
/mnt/nor/nnunet/raw_v2/Dataset104_TotalSegmentator_public_part4 \
class_map_part_muscles

python convert_dataset_to_nnunet.py \
/mnt/nor/wasserthalj_data/TotalSegmentator/zenodo_upload/Totalsegmentator_dataset \
/mnt/nor/nnunet/raw_v2/Dataset105_TotalSegmentator_public_part5 \
class_map_part_ribs
