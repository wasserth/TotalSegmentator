# Changes and improvements in TotalSegmentator v3

In v3 we did the following improvements:

* `total` and `appendicular_bones` tasks: We added 291 pediatric CT images to improve the segmentation performance on children. Moreover, we greatly improved the label quality: We fixed many small errors in the segmentations, we refined the labels especially for bones structures (e.g. femur segmentations a lot more precise now), and we fixed the vertebrae label mixups. This results in a more accurate segmentation model. Since the previous version was already quite good the changes might be noticable only for very specific cases.  

* `total_mr` and `appendicular_bones_mr` tasks: We added 246 whole body MR images, 249 pediatric MR images, and 170 MR images where the model failed before. Moreover, we improved the label quality on all existing cases. This results in a greatly improved segmentation model.

* `total_highres` task: The `total` model is trained on images resampled to 1.5mm isotropic resolution which is a good trade-off between speed and accuracy for most use cases. However, for some cases a higher resolution is beneficial. For these cases you can use the `total_highres` model which is trained on images resampled to 0.75 x 0.75 x 1.0mm isotropic resolution. But be aware: This takes a lot more RAM, GPU memory and runtime! Do not run on big CT images.

If you need the same segmentation results as in v2, you can use the task `total_v2` and `total_mr_v2`. This runs the old models from v2.


## Breaking changes from v2 to v3
* `total` task: class 26 is vertebrae_L6 instead of vertebrae_S1. So most of the time vertebrae_L6 is empty since most people on have 5 lumbar vertebrae. `sacrum` contains the full sacrum now. In the past `vertebrae_S1` contained parts of the sacrum and contained L6 if present. The new behaviour is more intuitive.
* (for `total_mr` everything stayed the same)


## Speed improvements
* We added a new option `--model_size small` which uses a model with less feature maps. This reduces runtime and memory usage. On GPU this makes less of a difference since runtime is dominated by pre/postprocessing. But on CPU this increases runtime by 2.5x. `--model_size small` leads to reduced accuracy but is still better than using `--fast` which uses a lower-resolution model.

Measured runtime, RAM and GPU memory for these options (and for `total_highres`) are in [runtime.md](runtime.md).



## Public dataset

The public training dataset for `total` and `total_mr` was updated to contain these new images and labels.

For v3 we did not use a `validation` set anymore. There is only `train` and `test` set.


## Still open problems

* the start of the ribs close to the spine is always missing a small part
* the GT segmentation of the colon/small_bowel is sometimes bad because the colon is so messy it is not possible to disentangle colon and small_bowel
