# Guide how to train your own nnU-Net on the TotalSegmentator dataset

1. Setup nnU-Net as described [here](https://github.com/MIC-DKFZ/nnUNet)
2. Download the data
3. Convert the data to nnU-Net format using `resources/convert_dataset_to_nnunet.py` (see `resources/train_nnunet.sh` for usage example)
4. Preprocess `nnUNetv2_plan_and_preprocess -d <your_dataset_id> -pl ExperimentPlanner -c 3d_fullres -np 2`
5. Train `nnUNetv2_train <your_dataset_id> 3d_fullres 0 -tr nnUNetTrainerNoMirroring` (takes several days)
6. Predict test set `nnUNetv2_predict -i path/to/imagesTs -o path/to/labelsTs_predicted -d <your_dataset_id> -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0`
7. Evaluate `python resources/evaluate.py path/to/labelsTs path/to/labelsTs_predicted` (requires `pip install git+https://github.com/google-deepmind/surface-distance.git`). The resulting numbers should be similar to the ones in `resources/evaluate_results.txt` (since training is not deterministic the mean dice score across all classes can vary by up to one dice point)
8. Done

> Note: This will not give you the same results as TotalSegmentator for two reasons:
1. TotalSegmentator v2 uses a [bigger dataset](resources/improvements_in_v2.md) which is not completely public
2. TotalSegmentator is trained on images without blurred faces. Your dataset contains blurred faces.
