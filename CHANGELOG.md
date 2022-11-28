## Master
* bugfix in cucim resampling
* add 6mm body model
* multilabel files contain label names in extended header


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