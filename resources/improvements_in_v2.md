# Improvements in training data of TotalSegmentator v2

List of new classes:
* costal cartilages
* prostate
* thyroid
* TODO: add all

Some of these new classes were available in some preliminary version as additional models in v1. Now they are properly added.


List of classes where we corrected some systemic errors in the labels (e.g. myocardium always slightly misaligned) or corrected several errors (e.g. bleedings in the liver sometimes not correctly labeled). For these classes you can expect a bit better segmentations now:
* femur
* hip
* heart chambers
* aorta
* liver
* spleen


We increased the number of training images from 1139 to 1559. We added the following images:
* more whole body images where TotalSegmentator failed before 
* images of feet and hands  (these were not included so far)
* more images of the head
* more images with bleedings in the abdomen where TotalSegmentator failed before
* more images from GE scanners and other institutions


Still open problems:
* the start of the ribs close to the spine is always missing a small part
* the end of the ribs at the costal cartilages is not properly defined
* the GT segmentation of the colon/small_bowel is sometimes bad because the colon is so messy it is not possible to disentangle colon and small_bowel
* there might be other problems I am not aware of


## Changes/improvements for public dataset:
* XX new classes (all classes of the `total` task)
* improved label quality (see above) (better than before but probably still some errors)
* more meta data per image (scanner, pathology, ...)
* less intrusive defacing
* No more corrupted files
* example code for converting to nnU-Net format
* example code for evaluation
* same subjects as in v1 (we did not publish the additional subjects we used for TotalSegmentator v2 training)
* labels are additional tasks are not included

