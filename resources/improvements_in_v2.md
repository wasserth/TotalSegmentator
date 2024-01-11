# Changes and improvements in TotalSegmentator v2


## Breaking changes from v1 to v2
The order of the classes has changed in the multilabel output file. If you use the option `--ml` checkout the new order [here](https://github.com/wasserth/TotalSegmentator#class-details). You can use the option `--v1_order` to use the old order from v1. However, the results will not contain the new v2 classes then. The resulting segmentations will also be slightly different from v1, because all models have been retrained. The heart chambers and the face will also be empty since those moved to the subtasks `heartchambers_highres` and `face`.
Everything else should be identical.


## New classes (n=33)

List of new classes by task:

total:
```
skull, thyroid_gland, prostate, brachiocephalic_vein_left, brachiocephalic_vein_right, brachiocephalic_trunk, common_carotid_artery_left, common_carotid_artery_right, atrial_appendage_left, subclavian_artery_left, subclavian_artery_right, vertebrae_S1, sternum, costal_cartilages, pulmonary_vein, superior_vena_cava, kidney_cyst_left, kidney_cyst_right, spinal_cord
```

appendicular_bones:
```
patella, tibia, fibula, tarsal, metatarsal, phalanges_feet, ulna, radius, carpal, metacarpal, phalanges_hand
```

tissue_types:
```
subcutaneous_fat, skeletal_muscle, torso_fat
```

vertebrae_body:
```
vertebrae_body (vertebrae without the spinous process)
```

The following classes were moved from the `total` task to the `heartchambers_highres` task:
```
heart_myocardium, heart_atrium_left, heart_ventricle_left, heart_atrium_right, heart_ventricle_right, pulmonary_artery
```
`total` now only contains the overall class `heart` instead.

Some of these new classes were available in some preliminary version as additional tasks in v1. Now they are properly added.

The following tasks are freely available only for non-commercial usage (all other tasks can also be used commercially):
`appendicular_bones`, `tissue_types`, `face`, `heartchambers_highres`, `vertebrae_body`


## Speed improvements
* when using the option `--roi_subset` first a very low resolution model is run (very fast runtime) to locate the rois you have specified. Then the image is cropped to this region and the full resolution model is run. This can save a lot of runtime and memory especially on CPU. (e.g. in a fully body CT image you are only interested in the left kidney. By using `--roi_subset left_kidney` the runtime on GPU is 5x faster and on CPU 32x faster. Memory consumption is 40% lower.)


## Improvements in training dataset

List of classes where we corrected some systemic errors in the labels (e.g. myocardium always slightly misaligned) or corrected several errors (e.g. bleedings in the liver sometimes not correctly labeled). For these classes you can expect slightly better segmentations now:
* femur
* humerus
* hip
* heart chambers
* aorta
* liver
* spleen
* kidney

We increased the number of training images from 1139 to 1559. We added the following images:
* more whole body images where TotalSegmentator failed before
* images of feet and hands  (these were not included so far)
* more images of the head
* more images with bleedings in the abdomen where TotalSegmentator failed before
* more images from GE scanners and other institutions

> NOTE: The public dataset does not contain these additional subjects. However, it contains several improvements (see next section).


Still open problems:
* the start of the ribs close to the spine is always missing a small part
* the end of the ribs at the costal cartilages is not properly defined
* the GT segmentation of the colon/small_bowel is sometimes bad because the colon is so messy it is not possible to disentangle colon and small_bowel
* there might be other problems I am not aware of


## Changes/improvements for public dataset:
* increased number of classes from 104 to 117 (all classes of the `total` task)
* removed heart chamber classes
* improved label quality (see above) (better than before but probably still some errors)
* more meta data per image (scanner, pathology, ...)
* less intrusive defacing
* No more corrupted files
* example code for converting to nnU-Net format
* example code for evaluation
* same subjects as in v1 for training and validation (we did not publish the additional subjects we used for TotalSegmentator v2 training)
* a few more subjects than in v1 for testing
* NOTE: labels for additional tasks are not included

