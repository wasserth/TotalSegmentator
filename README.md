# TotalSegmentator

Tool for segmentation of 104 classes in CT images. It was trained on a wide range of different CT images (different scanners, institutions, protocols,...) and therefore should work well on most images.

![Alt text](resources/imgs/overview_classes.png)

If you use it please cite our paper: todo


### Installation

Install dependencies:  
* [Pytorch](http://pytorch.org/)
* You should not have any nnU-Net installation in your python environment since TotalSegmentator will install its own 
custom installation.

Install Totalsegmentator
```
pip install git+https://github.com/wasserth/TotalSegmentator.git
```


### Usage
```
TotalSegmentator -i ct.nii.gz -o segmentations --fast --preview
```
> Note: TotalSegmentator only works with a NVidia GPU. If you do not have one you can try our online tool: www.totalsegmentator.ai


### Advanced settings
* `--fast`: For faster runtime and less memory requirements use this option. It will run a lower resolution model (3mm instead of 1.5mm). 
* `--preview`: This will generate a 3D rendering of all classes, giving you a quick overview if the segmentation worked and where it failed (see `preview.png` in output directory).
* `--statistics`: This will generate a csv file with volume and mean intensity of each class.


### Resource Requirements
For a quite big CT image Totalsegmentator has the following memory requirements:  
(1.5mm is the normal model and 3mm is the `--fast` model)

![Alt text](resources/imgs/runtime_table.png)


### Other commands
If you want to combine some subclasses (e.g. lung lobes) into one binary mask (e.g. entire lung) you can use the following command:
```
totalseg_combine_masks -i totalsegmentator_output_dir -o combined_mask.nii.gz -m lung
```

### Reference 
For more details see this paper (TODO).
If you use this tool please cite the following paper
```
Wasserthal et al. TODO
```
