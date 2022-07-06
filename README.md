# TotalSegmentator

Tool for segmentation of 104 classes in CT images. It was trained on a wide range of different CT images (different scanners, institutions, protocols,...) and therefore should work well on most images.

![Alt text](resources/imgs/overview_classes.png)

If you use it please cite our paper: todo


### Installation

Install dependencies
```
install pytorch from pytorch.org
```

Install Totalsegmentator
```
pip install git+https://github.com/wasserth/TotalSegmentator.git
```

### Usage
```
TotalSegmentator -i ct.nii.gz -o segmentations --fast --preview
```

### List of classes
TODO: Insert video/gif
TODO: Add list


### Advanced settings
For faster runtime and less memory requirements you can use the option `--fast`. This will run a lower resolution model (3mm).
TotalSegmentator only works with a NVidia GPU. If you do not have one you can try our online tool: www.totalsegmentator.ai


### Resource Requirements
For a quite big CT image Totalsegmentator has the following memory requirements:  
(1.5mm is the normal model and 3mm is the `--fast` model)

![Alt text](resources/imgs/runtime_table.png)


### Reference 
For more details see this paper (TODO).
If you use this tool please cite the following paper
```
Wasserthal et al. TODO
```
