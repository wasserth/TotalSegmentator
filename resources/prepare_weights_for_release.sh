#!/bin/bash
set -e  # Exit on error

# Prepare weights for release by removing subject ids and unneeded files and zipping
#
# Usage: ./prepare_weights_for_release.sh DATASET_ID [DATASET_ID2 ...]
#
# Example: ./prepare_weights_for_release.sh 527 528 529 ...

# todo: select as needed
# cd /mnt/nvme/data/multiseg/weights_upload/totalsegmentator_v2
# cd /mnt/nvme/data/multiseg/weights_upload/totalsegmentator_mri
# cd /mnt/nvme/data/multiseg/weights_upload/nnunet_modal
cd /mnt/nvme/data/multiseg/weights_upload/totalsegmentator_v3

# Process each dataset ID
for dataset_id in "$@"; do
    # Get full task name using Python script
    task_name=$(python3 -c "
from totalsegmentator.nnunet import get_full_task_name_v2
print(get_full_task_name_v2($dataset_id))
")
    
    echo "Processing $task_name..."
    
    # Copy dataset folder
    # cp -r "$nnUNet_results/$task_name" .
    
    # Anonymize pkl files in every trainer folder
    for trainer_folder in "$task_name"/*; do
        if [ -d "$trainer_folder" ]; then
            echo "Anonymizing $trainer_folder..."
            python ~/dev/TotalSegmentator/resources/anonymise_nnunet_pkl_v2.py "$trainer_folder"
        fi
    done
    
    # Create one zip archive containing all trainers
    zip -r "${task_name}.zip" "$task_name"
    
    echo "Completed processing $task_name"
done

echo "All datasets processed successfully"

