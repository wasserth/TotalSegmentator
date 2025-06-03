## Master


## Release 2.9.0
* use `--higher_order_resampling` argument
* add `--save_probabilities` argument: save softmax probabilities. Very experimental. Requires python skills.
* bugfix in task `brain_structures`: resampling was in different order than model. resulted in slower runtime and higher RAM usage and slightly reduced accuracy.
* add `craniofacial_structures` task: add skull, mandible, teeth, sinus, etc.


## Release 2.8.0
* make `totalseg_get_modality` work with normalized intensities (images which do not have original HU values anymore)
* replace pkg_resources for python 3.12 compatibility
* remove torch < 2.6.0 requirement (fixed in nnU-Net)
* add `--robust_crop`
* improved `tissue_types_mr` model
* add mode `nearest` to ndimage.zoom to fix resampling [bug](https://github.com/wasserth/TotalSegmentator/issues/458) (thanks to @baderstine for spotting this bug)


## Release 2.7.0
* add `liver_segments` and `liver_segments_mr` tasks


## Release 2.6.0
* add evans index calculation
* require torch < 2.6.0 to avoid weight loading error (waiting for nnU-Net to fix this issue)


## Release 2.5.0
* **major update to all MR models**: double the number of training subjects (this includes a retraining of these models. Therefore results can be different from previous versions.)
* **breaking change**: the `total_mr` task now contains 50 main classes instead of 56. Some classes moved to some other tasks and `total_mr` now contains also some classes which were not part of it previously. Please see the new `total_mr` class map in the readme or in `totalsegmentator/map_to_binary.py`.
* add `tissue_4_types` task: add intermuscular fat class
* add `vertebrae_mr` task: numbered single vertebrae segmentation in MR images (for CT this is already part of the `total` task)
* add `appendicular_bones_mr` task: add appendicular bones segmentation for MR images
* add `thigh_shoulder_muscles_mr` task: add thigh and shoulder muscles segmentation for CT and MR images
* add `vertebrae_body` with new class `intervertebral_discs`
* add `body_mr` task: add body segmentation for MR images
* add `kidney_cysts` task: greatly improved kidney cyst segmentation compared to the `kidney_cyst` class which is part of `total` task
* add `breasts` task: add breast segmentation
* add `oculomotor_muscles` task: add oculomotor muscles model (thanks to Philippe)
* add `lung_nodules` task (thanks to [BLUEMIND AI](https://bluemind.co/))
* update `coronary_arteries` task: increased number of training subjects, including non-contrast images
* add option to remove small connected components in postprocessing
* add `totalseg_get_modality`: estimate modality (CT or MR) from input image
* removed `rt_utils` and `p_tqdm` dependency
* change pi_time threshold for arterial late phase from 50s to 60s


## Release 2.4.0
* add brain structures
* add liver vessels
* greatly improved phase classification model


## Release 2.3.0
* Bugfixes
* add headneck structures


## Release 2.2.1
* also return statistics from python api
* add `totalseg_get_phase`
* major bugfix: rib labels were in wrong order
* hide nnunetv2 2.3.1 warning: `Detected old nnU-Net plans format. Attempting to reconstruct network architecture...`
* add mr models


## Release 2.1.0
* Bugfix: add flush to DummyFile
* Require python >= 3.9 in setup.py
* properly add `vertebrae_body` model
* add `--roi_subset_robust` argument
* add `--fastest` argument
* allow `mps` as device (but not supported by pytorch yet)
* add inline python version requirement for `requests` package
* if input spacing same as resampling spacing then skip resampling
* from python api also return nifti with label map in header
* input to python api can be a Nifti1Image object or a file path
* upgrade to `nnunetv2>=2.2.1`
* for `total` task use nnU-Net `step_size=0.8` instead of `0.5` for faster runtime while only decreasing dice by 0.001
* minor edits and bugfixes


## Release 2.0.5
* downgrade nnunet to 2.1 to fix bug in `fast` model


## Release 2.0.4
* temporary fix of critical bug in `fast` model. Proper fix in next release.


## LEGACY BUGFIX Release 1.5.7
* download all weights from github releases instead of zenodo


## Release 2.0.3
* fix critical bug in `body` task postprocessing: sometimes all foreground removed


## Release 2.0.2
* allow more than 10 classes in `--roi_subset`
* bugfix in `appendicular_bones` auxiliary mapping
* in multilable output only show classes selected in `--roi_subset` if selected
* make statistics work with dicom input


## Release 2.0.1
* add option `--v1_order` to use the old class order from v1


## Release 2.0.0
* train models with nnU-Net v2 (nnunet_cust dependency no longer needed)
* roi_subset a lot faster, because cropping with 6mm low res model to roi first
* more classes and improved training dataset (for details see `resources/improvements_in_v2.md`)
* bugfix to make cli available on windows
* bugfixes in dicom io
* add `--skip_saving` argument
* automatic tests on windows, linux and mac
* statistics are not calculated anymore for ROIs which are cut off by the top or bottom of the image (use `stats_include_incomplete` to change this behaviour)
* add postprocessing for body segmentation: remove small blobs
* use dicom2nifti for dicom conversion instead of dcm2niix because easier to use across platforms


## Release 1.5.6
* remove verbose print outs not needed
* add helper script for manual setup
* add fast statistics
* download weights from different server for faster and more stable download
* fix `requests` version to avoid `urllib3` openssl error
* minor bugfixes


## Release 1.5.5
* add independent script to download weights
* bugfixes


## Release 1.5.4
* support dicom input
* support dicom rt struct output
* add usage stats


## Release 1.5.3
* Correct wording in error messages
* add `--roi_subset` argument
* Use newer nnunet-customized version to avoid sklearn import error
* add `totalseg_import_weights` function
* add python api


## Release 1.5.2
* bugfix in cucim resampling
* add 6mm body model
* multilabel files contain label names in extended header
* add body model
* add pleural effusion model
* remove SimpleITK version requirement


## Release 1.4.0
* bugfixes
* add lung_vessels model
* add intracerebral hemorrhage model
* add coronary artery model
* preview file was renamed from `preview.png` to `preview_total.png`
* Split very big images into 3 parts and process one by one to avoid memory problems
* fix: check if input is 4d and then truncate to 3d
* make it work with windows
* make it work with cpu


## Release 1.3
* make output spacing exactly match input spacing
* improve weights download


## Release 1.2
* fix SimpleITK version to 2.0.2 to avoid nifti loading error


## Release 1.1
* Optimise statistics runtime
* fix server bugs
* add radiomics feature calculation


## Release 1.0
* Initial release