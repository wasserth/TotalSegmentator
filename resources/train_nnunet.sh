#!/bin/bash
set -e

############ Convert data ############

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


############ Train model ############

nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerNoMirroring
nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainerNoMirroring
nnUNetv2_train 103 3d_fullres 0 -tr nnUNetTrainerNoMirroring
nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerNoMirroring
nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainerNoMirroring


############ Predict test set ############

cd /mnt/nor/nnunet/raw_v2/Dataset101_TotalSegmentator_public_part1
nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 101 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

cd /mnt/nor/nnunet/raw_v2/Dataset102_TotalSegmentator_public_part2
nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 102 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

cd /mnt/nor/nnunet/raw_v2/Dataset103_TotalSegmentator_public_part3
nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 103 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

cd /mnt/nor/nnunet/raw_v2/Dataset104_TotalSegmentator_public_part4
nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 104 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

cd /mnt/nor/nnunet/raw_v2/Dataset105_TotalSegmentator_public_part5
nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 105 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0


############ Calculate dice score ############

cd /mnt/nor/nnunet/raw_v2/Dataset101_TotalSegmentator_public_part1
python ~/dev/TotalSegmentator/resources/evaluate.py labelsTs labelsTs_predicted class_map_part_organs

cd /mnt/nor/nnunet/raw_v2/Dataset102_TotalSegmentator_public_part2
python ~/dev/TotalSegmentator/resources/evaluate.py labelsTs labelsTs_predicted class_map_part_vertebrae

cd /mnt/nor/nnunet/raw_v2/Dataset103_TotalSegmentator_public_part3
python ~/dev/TotalSegmentator/resources/evaluate.py labelsTs labelsTs_predicted class_map_part_cardiac

cd /mnt/nor/nnunet/raw_v2/Dataset104_TotalSegmentator_public_part4
python ~/dev/TotalSegmentator/resources/evaluate.py labelsTs labelsTs_predicted class_map_part_muscles

cd /mnt/nor/nnunet/raw_v2/Dataset105_TotalSegmentator_public_part5
python ~/dev/TotalSegmentator/resources/evaluate.py labelsTs labelsTs_predicted class_map_part_ribs
