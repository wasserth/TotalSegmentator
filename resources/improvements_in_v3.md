# Changes and improvements in TotalSegmentator v3

* `total` task: TODO
* `total_mr` task: TODO
* `appendicular_bones_mr` task: TODO
* `total_highres` task: TODO


## Breaking changes from v2 to v3
* `total` task: class 26 is vertebrae_L6 instead of vertebrae_S1. TODO EXPLAIN MORE
* (for `total_mr` everything stayed the same)


## Speed improvements
* TODO


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
* the GT segmentation of the colon/small_bowel is sometimes bad because the colon is so messy it is not possible to disentangle colon and small_bowel
