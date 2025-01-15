import os
import sys
import pickle
import shutil
from pathlib import Path


if __name__ == "__main__":
    """
    Remove elements with Subject IDs from nnunet results.

    Also deleting the training log because in there are also the IDs.

    usage:

    cd $RESULTS_FOLDER
    anonymise_nnunet_pkl_v2.py nnUNet/3d_fullres/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres
    nnUNet_export_model_to_zip -t 291 -o dataset_291_upload.zip -c 3d_fullres -tr nnUNetTrainerV2 --not_strict
    """
    dir_in = Path(sys.argv[1])

    folds = sorted(list(dir_in.glob("fold_*")))
    print(f"Nr of folds found: {len(folds)}")

    # Anonymise model_final_checkpoint.model.pkl for all folds
    for fold_dir in folds:

        # Remove unneeded files and directories
        for dir in [fold_dir / "validation"]:
            if dir.exists():
                print(f"Deleting: {dir}")
                shutil.rmtree(dir)
    
        for file in [fold_dir / "checkpoint_best.pth"]:
            if file.exists():
                print(f"Deleting: {file}")
                os.remove(file)

        training_logs = fold_dir.glob("training_log_*")
        for log in training_logs:
            if log.exists():
                print(f"Deleting: {log}")
                os.remove(log)
