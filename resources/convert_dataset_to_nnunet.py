import sys
import os
from pathlib import Path
import shutil

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

from totalsegmentator.map_to_binary import class_map_5_parts


def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, labels):
    """
    foldername: nnUNet dataset name
    labels: class names 
    train_all: make fold 0 have 100% of the data in the training data (good for very small datasets)
    """
    print("Creating dataset.json...")
    out_base = Path(os.environ['nnUNet_raw']) / foldername

    json_dict = {}
    json_dict['name'] = "TotalSegmentator"
    json_dict['description'] = "Segmentation of TotalSegmentator classes"
    json_dict['reference'] = "https://zenodo.org/record/6802614"
    json_dict['licence'] = "Apache 2.0"
    json_dict['release'] = "2.0"
    json_dict['channel_names'] = {"0": "CT"}
    json_dict['labels'] = {val:idx for idx,val in enumerate(["background",] + labels)}
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'

    save_json(json_dict, os.path.join(out_base, "dataset.json"), sort_keys=False)

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnUNet_preprocessed']) / foldername
    output_folder_pkl.mkdir(exist_ok=True)

    splits = []
    splits.append({
        "train": list(subjects_train),
        "val": list(subjects_val)
    })

    print(f"nr of folds: {len(splits)}")
    print(f"nr train subjects (fold 0): {len(splits[0]['train'])}")
    print(f"nr val subjects (fold 0): {len(splits[0]['val'])}")

    save_json(splits, output_folder_pkl / "splits_final.json", sort_keys=False)


def combine_labels(ref_img, file_out, masks):
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)  
        if file_in.exists():
            img = nib.load(file_in)
            combined[img.get_fdata() > 0] = idx+1
        else:
            print(f"Missing: {file_in}")

    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


if __name__ == "__main__":    
    """
    Convert the downloaded TotalSegmentator dataset to nnUNet format and generate dataset.json and splits_final.json

    example usage: 
    python convert_dataset_to_nnunet.py /my_downloads/TotalSegmentator_dataset /nnunet/raw/Dataset100_TotalSegmentator_part1

    You must set nnUNet_raw and nnUNet_preprocessed environment variables before running this (see nnUNet documentation).
    """

    # TotalSegmentator is made up of 5 models. Choose which one you want to produce.
    class_map = class_map_5_parts["class_map_part_organs"]
    # class_map = class_map_5_parts["class_map_part_vertebrae"]
    # class_map = class_map_5_parts["class_map_part_cardiac"]
    # class_map = class_map_5_parts["class_map_part_muscles"]
    # class_map = class_map_5_parts["class_map_part_ribs"]

    dataset_path = Path(sys.argv[1])  # directory containining all the subjects
    nnunet_path = Path(sys.argv[2])  # directory of the new nnunet dataset

    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(dataset_path / "meta.csv")
    subjects_train = meta[meta["split"] == "train"]["image_id"].values
    subjects_val = meta[meta["split"] == "val"]["image_id"].values
    subjects_test = meta[meta["split"] == "test"]["image_id"].values

    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTr" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
                       nnunet_path / "labelsTr" / f"{subject}.nii.gz",
                       [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    for subject in tqdm(subjects_test):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
                       nnunet_path / "labelsTs" / f"{subject}.nii.gz",
                       [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    generate_json_from_dir_v2(nnunet_path.name, subjects_train, subjects_val, class_map.values())


"""
nnUNet commands for preprocessing and training:

nnUNetv2_plan_and_preprocess -d <your_dataset_id> -pl ExperimentPlanner -c 3d_fullres
nnUNetv2_train <your_dataset_id> 3d_fullres 0 -tr nnUNetTrainer_4000epochs_NoMirroring

After training:
use nnUNet_predict to predict the images in imagesTs to a directory called labelsTs_predicted

Then run evaluation:
python totalsegmentator/resources/evaluate.py path_to_labelsTs path_to_labelsTs_predicted 
"""
