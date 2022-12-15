# TotalSegmentator

Tool for segmentation of 104 classes in CT images. It was trained on a wide range of different CT images (different scanners, institutions, protocols,...) and therefore should work well on most images. The training dataset with 1204 subjects can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.6802613). You can also try the tool online at [totalsegmentator.com](https://totalsegmentator.com/).

![Alt text](resources/imgs/overview_classes.png)

Created by the department of [Research and Analysis at University Hospital Basel](https://www.unispital-basel.ch/en/radiologie-nuklearmedizin/forschung).  
If you use it please cite our paper: [https://arxiv.org/abs/2208.05868](https://arxiv.org/abs/2208.05868).  



### Installation

TotalSegmentator works on Ubuntu, Mac and Windows and on CPU and GPU (on CPU it is slow).

Install dependencies:  
* Python >= 3.7
* [Pytorch](http://pytorch.org/)
* if you use the option `--preview` you have to install xvfb (`apt-get install xvfb`)
* You should not have any nnU-Net installation in your python environment since TotalSegmentator will install its own custom installation.

* optionally: for faster resampling you can use `cucim` (`pip install cupy-cuda11x cucim`)

Install Totalsegmentator
```
pip install TotalSegmentator
```


### Usage
```
TotalSegmentator -i ct.nii.gz -o segmentations
```
> Note: Only nifti files are supported. To convert dicom files to nifti we recommend [dcm2niix](https://github.com/rordenlab/dcm2niix).  

> Note: If a CUDA compatible GPU is available TotalSegmentator will automatically use it. Otherwise it will use the CPU, which is a lot slower and should only be used with the `--fast` option.  

> Note: You can also try it online: [www.totalsegmentator.com](https://totalsegmentator.com/) (supports dicom files)


### Advanced settings
* `--fast`: For faster runtime and less memory requirements use this option. It will run a lower resolution model (3mm instead of 1.5mm). 
* `--preview`: This will generate a 3D rendering of all classes, giving you a quick overview if the segmentation worked and where it failed (see `preview.png` in output directory).
* `--statistics`: This will generate a file `statistics.json` with volume (in mm³) and mean intensity of each class.
* `--radiomics`: This will generate a file `statistics_radiomics.json` with radiomics features of each class. You have to install pyradiomics to use this (`pip install pyradiomics`).


### Run via docker
We also provide a docker container which can be used the following way
```
docker run --gpus 'device=0' --ipc=host -v /absolute/path/to/my/data/directory:/tmp wasserth/totalsegmentator_container:master TotalSegmentator -i /tmp/ct.nii.gz -o /tmp/segmentations
```

### Subtasks
We added some more models to TotalSegmentator beyond the default one. This allows segmentation of even 
more classes in more detailed subparts of the image. First you have to run TotalSegmentator with the 
normal settings to get the normal masks. These masks are required to crop the image to a subregion on 
which the detailed model will run.
This is only available in the latest master branch at the moment.
```
TotalSegmentator -i ct.nii.gz -o segmentations --fast
TotalSegmentator -i ct.nii.gz -o segmentations -ta lung_vessels
TotalSegmentator -i ct.nii.gz -o segmentations -ta cerebral_bleed
```

### Resource Requirements
Totalsegmentator has the following runtime and memory requirements (using a Nvidia RTX 3090 GPU):  
(1.5mm is the normal model and 3mm is the `--fast` model)

![Alt text](resources/imgs/runtime_table.png)


### Train / validation / test split
The exact split of the dataset can be found in the file `meta.csv` inside of the [dataset](https://doi.org/10.5281/zenodo.6802613). This was used for the validation in our paper.  
The exact numbers of the results for the high resolution model (1.5mm) can be found [here](resources/results_all_classes.json). The paper shown these numbers in the supplementary materials figure 11.


### Retrain model on your own
You have to download the data and then follow the instructions of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) how to train a nnU-Net. We trained a `3d_fullres` model and the only adaptation to the default training is setting the number of epochs to 4000 and deactivating mirror data augmentation. The adapted trainer can be found [here](https://github.com/wasserth/nnUNet_cust/blob/working_2022_03_18/nnunet/training/network_training/custom_trainers/nnUNetTrainerV2_ep4000_nomirror.py).
For combining the single masks into one multilabel file you can use the function `combine_masks_to_multilabel_file` in [totalsegmentator.libs](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/libs.py).


### Other commands
If you want to combine some subclasses (e.g. lung lobes) into one binary mask (e.g. entire lung) you can use the following command:
```
totalseg_combine_masks -i totalsegmentator_output_dir -o combined_mask.nii.gz -m lung
```

### Install latest master branch (contains latest bug fixes)
```
pip install git+https://github.com/wasserth/TotalSegmentator.git
```

### Reference 
For more details see this paper [https://arxiv.org/abs/2208.05868](https://arxiv.org/abs/2208.05868).
If you use this tool please cite it as follows
```
Wasserthal J., Meyer M., Breit H., Cyriac J., Yang S., Segeroth M. TotalSegmentator: robust segmentation of 104 anatomical structures in CT images, 2022. URL: https://arxiv.org/abs/2208.05868.  arXiv: 2208.05868
```
Moreover, we would really appreciate if you let us know what you are using this tool for. You can also tell us what classes we should add in future releases. You can do so [here](https://github.com/wasserth/TotalSegmentator/issues/1)..