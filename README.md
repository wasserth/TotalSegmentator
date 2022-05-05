# TotalSegmentator

Tool for segmentation of X classes in CT images. It was trained on a wide range of different CT images (different scanners, institutions, protocols,...) and therefore should work well on most images.

### Installation

Install dependencies
```
pip install batchgenerators==0.21                             
pip install https://github.com/wasserth/nnUNet_cust/archive/refs/heads/working_2022_03_18.zip
```

Install Totalsegmentator
```
pip install git+https://github.com/wasserth/TotalSegmentator.git
```

Copy nnUNet weights to
```
~/.totalsegmentator/nnunet/results/nnUNet/3d_fullres
```

### Usage
```
TotalSegmentator -i ct.nii.gz -o segmentations --fast --preview
```

### List of classes
TODO: Insert video/gif
TODO: Add list

### Reference 
For more details see this paper (TODO).
If you use this tool please cite the following paper
```
Wasserthal et al. TODO
```

### Advanced settings
For faster runtime you can use the option TODO.
For usage without a CPU we recommend the following settings for reasonable runtime: TODO.


### Resource Requirements
For a quite big CT image Totalsegmentator has the following memory requirements:  
with option `--fast` (using 3mm model):  

input 3mm image:  
mode: normal: GPU: roughly 5GB, RAM: 15.3GB, runtime 130s
mode: fastest: GPU: roughly 10GB, RAM: 7.0GB, runtime 25s

input 1.5mm image:
mode: normal: GPU: roughly 5GB, RAM: 15.6GB, runtime 140s
mode: fastest: GPU: roughly 10GB, RAM: 7.3GB, runtime 48s

