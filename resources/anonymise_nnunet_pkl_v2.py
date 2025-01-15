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

    cd $nnUNet_results
    anonymise_nnunet_pkl_v2.py Dataset789_kidney_cyst_501subj/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres
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
