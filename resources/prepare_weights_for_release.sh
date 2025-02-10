#!/bin/bash
set -e  # Exit on error

# Prepare weights for release by removing subject ids and unneeded files and zipping
#
# Usage: ./prepare_weights_for_release.sh DATASET_ID [DATASET_ID2 ...]
#
# Example: ./prepare_weights_for_release.sh 527 528 529 ...

# todo: select as needed
cd /mnt/nvme/data/multiseg/weights_upload/totalsegmentator_v2
# cd /mnt/nvme/data/multiseg/weights_upload/totalsegmentator_mri
# cd /mnt/nvme/data/multiseg/weights_upload/nnunet_modal

# Process each dataset ID
for dataset_id in "$@"; do
    # Get full task name using Python script
    task_name=$(python3 -c "
from totalsegmentator.nnunet import get_full_task_name_v2
print(get_full_task_name_v2($dataset_id))
")
    
    echo "Processing $task_name..."
    
    # Copy dataset folder
    cp -r "$nnUNet_results/$task_name" .
    
    # Get the only folder inside task_name
    trainer_folder=$(ls "$task_name" | head -n 1)
    
    # Anonymize the pkl files
    python ~/dev/TotalSegmentator/resources/anonymise_nnunet_pkl_v2.py "$task_name/$trainer_folder"
    
    # Create zip archive
    zip -r "${task_name}.zip" "$task_name"
    
    echo "Completed processing $task_name"
done

echo "All datasets processed successfully"

