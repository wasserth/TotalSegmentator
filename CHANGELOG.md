## Master

## Release 1.5.7
* download all weights from github releases instead of zenodo


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