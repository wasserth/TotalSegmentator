# TotalSegmentator

Tool for segmentation of most major anatomical structures in any CT or MR image. It was trained on a wide range of different CT and MR images (different scanners, institutions, protocols,...) and therefore works well on most images. A large part of the training dataset can be downloaded here: [CT dataset](https://doi.org/10.5281/zenodo.6802613) (1228 subjects) and [MR dataset](https://zenodo.org/doi/10.5281/zenodo.11367004) (616 subjects). You can also try the tool online at [totalsegmentator.com](https://totalsegmentator.com/) or as [3D Slicer extension](https://github.com/lassoan/SlicerTotalSegmentator).

**ANNOUNCEMENT: We created a platform where anyone can help annotate more data to further improve TotalSegmentator: [TotalSegmentator Annotation Platform](https://annotate.totalsegmentator.com).**  
  
**ANNOUNCEMENT: We created web applications for [abdominal organ volume](https://compute.totalsegmentator.com/volume-report/), [Evans index](https://compute.totalsegmentator.com/evans-index/), and [aorta diameter](https://compute.totalsegmentator.com/aorta-report/).**

Main classes for CT and MR:
![Alt text](resources/imgs/overview_classes_v2.png)

TotalSegmentator supports a lot more structures. See [subtasks](#subtasks) or [here](https://backend.totalsegmentator.com/find-task/) for more details.

Created by the department of [Research and Analysis at University Hospital Basel](https://www.unispital-basel.ch/en/radiologie-nuklearmedizin/forschung-radiologie-nuklearmedizin).
If you use it please cite our [Radiology AI paper](https://pubs.rsna.org/doi/10.1148/ryai.230024) ([free preprint](https://arxiv.org/abs/2208.05868)). If you use it for MR images please cite the [TotalSegmentator MRI Radiology paper](https://pubs.rsna.org/doi/10.1148/radiol.241613) ([free preprint](https://arxiv.org/abs/2405.19492)). Please also cite [nnUNet](https://github.com/MIC-DKFZ/nnUNet) since TotalSegmentator is heavily based on it.


### Installation

TotalSegmentator works on Ubuntu, Mac, and Windows and on CPU and GPU.

Install dependencies:
* Python >= 3.9
* [PyTorch](http://pytorch.org/) >= 2.0.0 and <2.6.0 (and <2.4 for Windows)

Optionally:
* if you use the option `--preview` you have to install xvfb (`apt-get install xvfb`) and fury (`pip install fury`)


Install Totalsegmentator
```bash
pip install TotalSegmentator
```


### Usage
For CT images:
```bash
TotalSegmentator -i ct.nii.gz -o segmentations
```
For MR images:
```bash
TotalSegmentator -i mri.nii.gz -o segmentations --task total_mr
```
> Note: A Nifti file or a folder (or zip file) with all DICOM slices of one patient is allowed as input.

> Note: If you run on CPU use the option `--fast` or `--roi_subset` to greatly improve runtime.

> Note: This is not a medical device and is not intended for clinical usage. However, it is part of several FDA-approved products, where it has been certified as a component of the overall system.


### Subtasks

![Alt text](resources/imgs/overview_subclasses_2.png)

Next to the default task (`total`) there are more subtasks with more classes. If the taskname ends with `_mr` it works for MR images, otherwise for CT images.

Openly available for any usage (Apache-2.0 license):
* **total**: default task containing 117 main classes (see [here](https://github.com/wasserth/TotalSegmentator#class-details) for a list of classes)
* **total_mr**: default task containing 50 main classes on MR images (see [here](https://github.com/wasserth/TotalSegmentator#class-details) for a list of classes)
* **lung_vessels**: lung_vessels (cite [paper](https://www.sciencedirect.com/science/article/pii/S0720048X22001097)), lung_trachea_bronchia
* **body**: body, body_trunc, body_extremities, skin
* **body_mr**: body_trunc, body_extremities (for MR images)
* **vertebrae_mr**: sacrum, vertebrae_L5, vertebrae_L4, vertebrae_L3, vertebrae_L2, vertebrae_L1, vertebrae_T12, vertebrae_T11, vertebrae_T10, vertebrae_T9, vertebrae_T8, vertebrae_T7, vertebrae_T6, vertebrae_T5, vertebrae_T4, vertebrae_T3, vertebrae_T2, vertebrae_T1, vertebrae_C7, vertebrae_C6, vertebrae_C5, vertebrae_C4, vertebrae_C3, vertebrae_C2, vertebrae_C1 (for CT this is part of the `total` task)
* **cerebral_bleed**: intracerebral_hemorrhage (cite [paper](https://www.mdpi.com/2077-0383/12/7/2631))*
* **hip_implant**: hip_implant*
* **pleural_pericard_effusion**: pleural_effusion (cite [paper](http://dx.doi.org/10.1097/RLI.0000000000000869)), pericardial_effusion (cite [paper](http://dx.doi.org/10.3390/diagnostics12051045))*
* **head_glands_cavities**: eye_left, eye_right, eye_lens_left, eye_lens_right, optic_nerve_left, optic_nerve_right, parotid_gland_left, parotid_gland_right, submandibular_gland_right, submandibular_gland_left, nasopharynx, oropharynx, hypopharynx, nasal_cavity_right, nasal_cavity_left, auditory_canal_right, auditory_canal_left, soft_palate, hard_palate (cite [paper](https://www.mdpi.com/2072-6694/16/2/415))
* **head_muscles**: masseter_right, masseter_left, temporalis_right, temporalis_left, lateral_pterygoid_right, lateral_pterygoid_left, medial_pterygoid_right, medial_pterygoid_left, tongue, digastric_right, digastric_left
* **headneck_bones_vessels**: larynx_air, thyroid_cartilage, hyoid, cricoid_cartilage, zygomatic_arch_right, zygomatic_arch_left, styloid_process_right, styloid_process_left, internal_carotid_artery_right, internal_carotid_artery_left, internal_jugular_vein_right, internal_jugular_vein_left (cite [paper](https://www.mdpi.com/2072-6694/16/2/415))
* **headneck_muscles**: sternocleidomastoid_right, sternocleidomastoid_left, superior_pharyngeal_constrictor, middle_pharyngeal_constrictor, inferior_pharyngeal_constrictor, trapezius_right, trapezius_left, platysma_right, platysma_left, levator_scapulae_right, levator_scapulae_left, anterior_scalene_right, anterior_scalene_left, middle_scalene_right, middle_scalene_left, posterior_scalene_right, posterior_scalene_left, sterno_thyroid_right, sterno_thyroid_left, thyrohyoid_right, thyrohyoid_left, prevertebral_right, prevertebral_left (cite [paper](https://www.mdpi.com/2072-6694/16/2/415))
* **liver_vessels**: liver_vessels, liver_tumor (cite [paper](https://arxiv.org/abs/1902.09063))*
* **oculomotor_muscles**: skull, eyeball_right, lateral_rectus_muscle_right, superior_oblique_muscle_right, levator_palpebrae_superioris_right, superior_rectus_muscle_right, medial_rectus_muscle_left, inferior_oblique_muscle_right, inferior_rectus_muscle_right, optic_nerve_left, eyeball_left, lateral_rectus_muscle_left, superior_oblique_muscle_left, levator_palpebrae_superioris_left, superior_rectus_muscle_left, medial_rectus_muscle_right, inferior_oblique_muscle_left, inferior_rectus_muscle_left, optic_nerve_right*
* **lung_nodules**: lung, lung_nodules (provided by [BLUEMIND AI](https://bluemind.co/): Fitzjalen R., Aladin M., Nanyan G.) (trained on 1353 subjects, partly from LIDC-IDRI)
* **kidney_cysts**: kidney_cyst_left, kidney_cyst_right (strongly improved accuracy compared to kidney_cysts inside of `total` task)
* **breasts**: breast
* **liver_segments**: liver_segment_1, liver_segment_2, liver_segment_3, liver_segment_4, liver_segment_5, liver_segment_6, liver_segment_7, liver_segment_8 (Couinaud segments) (cite [paper](https://doi.org/10.1007/978-3-030-32692-0_32))*
* **liver_segments_mr**: liver_segment_1, liver_segment_2, liver_segment_3, liver_segment_4, liver_segment_5, liver_segment_6, liver_segment_7, liver_segment_8 (for MR images) (Couinaud segments)*
* **craniofacial_structures**: mandible, teeth_lower, skull, head, sinus_maxillary, sinus_frontal, teeth_upper
* **abdominal_muscles**: pectoralis_major_right, pectoralis_major_left, rectus_abdominis_right, rectus_abdominis_left, serratus_anterior_right, serratus_anterior_left, latissimus_dorsi_right, latissimus_dorsi_left, trapezius_right, trapezius_left, external_oblique_right, external_oblique_left, internal_oblique_right, internal_oblique_left, erector_spinae_right, erector_spinae_left, transversospinalis_right, transversospinalis_left, psoas_major_right, psoas_major_left, quadratus_lumborum_right, quadratus_lumborum_left (cite [paper](https://doi.org/10.1101/2025.01.13.25319967)) (only segments within T4-L4)*
* **teeth**: "lower_jawbone", "upper_jawbone", "left_inferior_alveolar_canal", "right_inferior_alveolar_canal", "left_maxillary_sinus", "right_maxillary_sinus", "pharynx", "bridge", "crown", "implant", "upper_right_central_incisor_fdi11", "upper_right_lateral_incisor_fdi12", "upper_right_canine_fdi13", "upper_right_first_premolar_fdi14", "upper_right_second_premolar_fdi15", "upper_right_first_molar_fdi16", "upper_right_second_molar_fdi17", "upper_right_third_molar_fdi18", "upper_left_central_incisor_fdi21", "upper_left_lateral_incisor_fdi22", "upper_left_canine_fdi23", "upper_left_first_premolar_fdi24", "upper_left_second_premolar_fdi25", "upper_left_first_molar_fdi26", "upper_left_second_molar_fdi27", "upper_left_third_molar_fdi28", "lower_left_central_incisor_fdi31", "lower_left_lateral_incisor_fdi32", "lower_left_canine_fdi33", "lower_left_first_premolar_fdi34", "lower_left_second_premolar_fdi35", "lower_left_first_molar_fdi36", "lower_left_second_molar_fdi37", "lower_left_third_molar_fdi38", "lower_right_central_incisor_fdi41", "lower_right_lateral_incisor_fdi42", "lower_right_canine_fdi43", "lower_right_first_premolar_fdi44", "lower_right_second_premolar_fdi45", "lower_right_first_molar_fdi46", "lower_right_second_molar_fdi47", "lower_right_third_molar_fdi48", "left_mandibular_incisive_canal_fdi103", "right_mandibular_incisive_canal_fdi104", "lingual_canal", "upper_right_central_incisor_pulp_fdi111", "upper_right_lateral_incisor_pulp_fdi112", "upper_right_canine_pulp_fdi113", "upper_right_first_premolar_pulp_fdi114", "upper_right_second_premolar_pulp_fdi115", "upper_right_first_molar_pulp_fdi116", "upper_right_second_molar_pulp_fdi117", "upper_right_third_molar_pulp_fdi118", "upper_left_central_incisor_pulp_fdi121", "upper_left_lateral_incisor_pulp_fdi122", "upper_left_canine_pulp_fdi123", "upper_left_first_premolar_pulp_fdi124", "upper_left_second_premolar_pulp_fdi125", "upper_left_first_molar_pulp_fdi126", "upper_left_second_molar_pulp_fdi127", "upper_left_third_molar_pulp_fdi128", "lower_left_central_incisor_pulp_fdi131", "lower_left_lateral_incisor_pulp_fdi132", "lower_left_canine_pulp_fdi133", "lower_left_first_premolar_pulp_fdi134", "lower_left_second_premolar_pulp_fdi135", "lower_left_first_molar_pulp_fdi136", "lower_left_second_molar_pulp_fdi137", "lower_left_third_molar_pulp_fdi138", "lower_right_central_incisor_pulp_fdi141", "lower_right_lateral_incisor_pulp_fdi142", "lower_right_canine_pulp_fdi143", "lower_right_first_premolar_pulp_fdi144", "lower_right_second_premolar_pulp_fdi145", "lower_right_first_molar_pulp_fdi146", "lower_right_second_molar_pulp_fdi147", "lower_right_third_molar_pulp_fdi148" (based on the ToothFairy3 dataset, cite [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.html))

*: These models are not trained on the full totalsegmentator dataset but on some small other datasets. Therefore, expect them to work less robustly.

Available with a license (free licenses available for non-commercial usage [here](https://backend.totalsegmentator.com/license-academic/). For a commercial license contact jakob.wasserthal@usb.ch):
* **heartchambers_highres**: myocardium, atrium_left, ventricle_left, atrium_right, ventricle_right, aorta, pulmonary_artery (trained on sub-millimeter resolution)
* **appendicular_bones**: patella, tibia, fibula, tarsal, metatarsal, phalanges_feet, ulna, radius, carpal, metacarpal, phalanges_hand
* **appendicular_bones_mr**: patella, tibia, fibula, tarsal, metatarsal, phalanges_feet, ulna, radius (for MR images)
* **tissue_types**: subcutaneous_fat, torso_fat, skeletal_muscle
* **tissue_types_mr**: subcutaneous_fat, torso_fat, skeletal_muscle (for MR images)
* **tissue_4_types**: subcutaneous_fat, torso_fat, skeletal_muscle, intermuscular_fat (in contrast to `tissue_types` skeletal_muscle is split into two classes: muscle and fat)
* **brain_structures**: brainstem, subarachnoid_space, venous_sinuses, septum_pellucidum, cerebellum, caudate_nucleus, lentiform_nucleus, insular_cortex, internal_capsule, ventricle, central_sulcus, frontal_lobe, parietal_lobe, occipital_lobe, temporal_lobe, thalamus (NOTE: this is for CT) (cite [paper](https://doi.org/10.1148/ryai.2020190183) as our model is partly based on this)
* **vertebrae_body**: vertebral body of all vertebrae (without the vertebral arch), intervertebral_discs (for MR this is part of the `total_mr` task)
* **face**: face_region (for anonymization)
* **face_mr**: face_region (for anonymization)
* **thigh_shoulder_muscles**: quadriceps_femoris_left, quadriceps_femoris_right, thigh_medial_compartment_left, thigh_medial_compartment_right, thigh_posterior_compartment_left, thigh_posterior_compartment_right, sartorius_left, sartorius_right, deltoid, supraspinatus, infraspinatus, subscapularis, coracobrachial, trapezius, pectoralis_minor, serratus_anterior, teres_major, triceps_brachii
* **thigh_shoulder_muscles_mr**: quadriceps_femoris_left, quadriceps_femoris_right, thigh_medial_compartment_left, thigh_medial_compartment_right, thigh_posterior_compartment_left, thigh_posterior_compartment_right, sartorius_left, sartorius_right, deltoid, supraspinatus, infraspinatus, subscapularis, coracobrachial, trapezius, pectoralis_minor, serratus_anterior, teres_major, triceps_brachii (for MR images)
* **coronary_arteries**: coronary_arteries (also works on non-contrast images)

Usage:
```bash
TotalSegmentator -i ct.nii.gz -o segmentations -ta <task_name>
```

Confused by all the structures and tasks? Check [this](https://backend.totalsegmentator.com/find-task/) to search through available structures and tasks.

The mapping from label ID to class name can be found [here](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py).

If you have a nnU-Net model for some structures not supported yet, you can contribute it. This will enable all TotalSegmentator users to easily use it and at the same time increase the reach of your work by more people citing your paper. Contact jakob.wasserthal@usb.ch.

Thank you to [INGEDATA](https://www.ingedata.ai/) for providing a team of radiologists to support some of the data annotations.


### Advanced settings
* `--device`: Choose `cpu` or `gpu` or `gpu:X (e.g., gpu:1 -> cuda:1)`
* `--fast`: For faster runtime and less memory requirements use this option. It will run a lower resolution model (3mm instead of 1.5mm).
* `--roi_subset`: Takes a space-separated list of class names (e.g. `spleen colon brain`) and only predicts those classes. Saves a lot of runtime and memory. Might be less accurate especially for small classes (e.g. prostate).
* `--robust_crop`: For some tasks and for roi_subset a 6mm low resolution model is used to crop to the region of interest. Sometimes this model is incorrect, which leads to artifacts like segmentations being cut off. robust_crop will use a better but slower 3mm model instead.
* `--preview`: This will generate a 3D rendering of all classes, giving you a quick overview if the segmentation worked and where it failed (see `preview.png` in output directory).
* `--ml`: This will save one nifti file containing all labels instead of one file for each class. Saves runtime during saving of nifti files. (see [here](https://github.com/wasserth/TotalSegmentator#class-details) for index to class name mapping).
* `--statistics`: This will generate a file `statistics.json` with volume (in mm³) and mean intensity of each class.
* `--radiomics`: This will generate a file `statistics_radiomics.json` with the radiomics features of each class. You have to install pyradiomics to use this (`pip install pyradiomics`).


### Other commands
If you want to know which contrast phase a CT image is you can use the following command (requires `pip install xgboost`). More details can be found [here](resources/contrast_phase_prediction.md):
```bash
totalseg_get_phase -i ct.nii.gz -o contrast_phase.json
```

If you want to know which modality (CT or MR) an image is you can use the following command (requires `pip install xgboost`). 
```bash
totalseg_get_modality -i image.nii.gz -o modality.json
```

If you want to combine some subclasses (e.g. lung lobes) into one binary mask (e.g. entire lung) you can use the following command:
```bash
totalseg_combine_masks -i totalsegmentator_output_dir -o combined_mask.nii.gz -m lungcomm 
```

If you want to calculate the [Evans index](https://radiopaedia.org/articles/evans-index-2) you can use the following command:
```bash
totalseg_evans_index -i ct_skull.nii.gz -o evans_index.json -p evans_index.png
```

Normally weights are automatically downloaded when running TotalSegmentator. If you want to download the weights with an extra command (e.g. when building a docker container) use this:
```bash
totalseg_download_weights -t <task_name>
```
This will download them to `~/.totalsegmentator/nnunet/results`. You can change this path by doing `export TOTALSEG_HOME_DIR=/new/path/.totalsegmentator`. If your machine has no internet, then download on another machine with internet and copy `~/.totalsegmentator` to the machine without internet.

After acquiring a license number for the non-open tasks you can set it with the following command:
```bash
totalseg_set_license -l aca_12345678910
```

You can output the softmax probabilities. This will give you a `.npz` file you can load with numpy. The geometry
might not be identical to your input image. There will also be a `.pkl` output file with geometry
information. This does not work well for the `total` task since this is based on multiple models.
```bash
TotalSegmentator -i ct.nii.gz -o seg -ta lung_nodules --save_probabilities probs.npz
```

If you do not have internet access on the machine you want to run TotalSegmentator on:
1. Install TotalSegmentator [and set up the license] on a machine with internet.
2. Run TotalSegmentator for one subject on this machine. This will download the weights and save them to `~/.totalsegmentator`.
3. Copy the folder `~/.totalsegmentator` from this machine to the machine without internet.
4. TotalSegmentator should now work also on the machine without internet.


### Web applications
We provide the following web applications to easily process your images:
* [TotalSegmentator](https://totalsegmentator.com/): Run totalsegmentator on your own images via a simple web interface.
* [TotalSegmentator Annotation Platform](https://annotate.totalsegmentator.com/): Help annotate more data to further improve TotalSegmentator.
* [Volume Report](https://compute.totalsegmentator.com/volume-report/): Get the volume of abdominal organs + tissue und bone density. Also show percentile in population.
* [Evans Index](https://compute.totalsegmentator.com/evans-index/): Compute the Evans index.
* [Aorta Report](https://compute.totalsegmentator.com/aorta-report/): Analyse the diameter along the aorta.


### Run via docker
We also provide a docker container which can be used the following way
```bash
docker run --gpus 'device=0' --ipc=host -v /absolute/path/to/my/data/directory:/tmp wasserth/totalsegmentator:2.2.1 TotalSegmentator -i /tmp/ct.nii.gz -o /tmp/segmentations
```


### Resource Requirements
Totalsegmentator has the following runtime and memory requirements (using an Nvidia RTX 3090 GPU):
(1.5mm is the normal model and 3mm is the `--fast` model. With v2 the runtimes have increased a bit since
we added more classes.)

![Alt text](resources/imgs/runtime_table.png)

If you want to reduce memory consumption you can use the following options:
* `--fast`: This will use a lower-resolution model
* `--body_seg`: This will crop the image to the body region before processing it
* `--roi_subset <list of classes>`: This will only predict a subset of classes
* `--force_split`: This will split the image into 3 parts and process them one after another. (Do not use this for small images. Splitting these into even smaller images will result in a field of view which is too small.)
* `--nr_thr_saving 1`: Saving big images with several threads will take a lot of memory


### Python API
You can run totalsegmentator via Python:
```python
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

if __name__ == "__main__":
    # option 1: provide input and output as file paths
    totalsegmentator(input_path, output_path)
    
    # option 2: provide input and output as nifti image objects
    input_img = nib.load(input_path)
    output_img = totalsegmentator(input_img)
    nib.save(output_img, output_path)
```
You can see all available arguments [here](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/python_api.py). Running from within the main environment should avoid some multiprocessing issues.

The segmentation image contains the names of the classes in the extended header. If you want to load this additional header information you can use the following code (requires `pip install xmltodict`):
```python
from totalsegmentator.nifti_ext_header import load_multilabel_nifti

segmentation_nifti_img, label_map_dict = load_multilabel_nifti(image_path)
```


### Install latest master branch (contains latest bug fixes)
```bash
pip install git+https://github.com/wasserth/TotalSegmentator.git
```


### Train/validation/test split
The exact split of the dataset can be found in the file `meta.csv` inside of the [dataset](https://doi.org/10.5281/zenodo.6802613). This was used for the validation in our paper.
The exact numbers of the results for the high-resolution model (1.5mm) can be found [here](resources/results_all_classes_v1.json). The paper shows these numbers in the supplementary materials Figure 11.


### Retrain model and run evaluation
See [here](resources/train_nnunet.md) for more info on how to train a nnU-Net yourself on the TotalSegmentator dataset, how to split the data into train/validation/test set as in our paper, and how to run the same evaluation as in our paper.


### Typical problems

**ITK loading Error**
When you get the following error message
```text
ITK ERROR: ITK only supports orthonormal direction cosines. No orthonormal definition was found!
```
you should do
```bash
pip install SimpleITK==2.0.2
```

Alternatively you can try
```bash
fslorient -copysform2qform input_file
[fslreorient2std input_file output_file]
```
or use [this python command](https://github.com/MIC-DKFZ/nnDetection/issues/24#issuecomment-2627684467).

**Bad segmentations**
When you get bad segmentation results check the following:
* does your input image contain the original HU values or are the intensity values rescaled to a different range?
* is the patient normally positioned in the image? (In axial view is the spine at the bottom of the image? In the coronal view is the head at the top of the image?)


### Running v1
If you want to keep on using TotalSegmentator v1 (e.g. because you do not want to change your pipeline) you
can install it with the following command:
```bash
pip install TotalSegmentator==1.5.7
```
The documentation for v1 can be found [here](https://github.com/wasserth/TotalSegmentator/tree/v1.5.7). Bugfixes for v1 are developed in the branch `v1_bugfixes`.
Our Radiology AI publication refers to TotalSegmentator v1.


### Other
* TotalSegmentator sends anonymous usage statistics to help us improve it further. You can deactivate it by setting `send_usage_stats` to `false` in `~/.totalsegmentator/config.json`.
* At [changes and improvements](resources/improvements_in_v2.md) you can see an overview of differences between v1 and v2.


### Reference
For more details see our [Radiology AI paper](https://pubs.rsna.org/doi/10.1148/ryai.230024) ([freely available preprint](https://arxiv.org/abs/2208.05868)).
If you use this tool please cite it as follows
```text
Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024
```
Please also cite [nnUNet](https://github.com/MIC-DKFZ/nnUNet) since TotalSegmentator is heavily based on it.
Moreover, we would really appreciate it if you let us know what you are using this tool for. You can also tell us what classes we should add in future releases. You can do so [here](https://github.com/wasserth/TotalSegmentator/issues/1).


### Class details

The following table shows a list of all classes for task `total`.

TA2 is a standardized way to name anatomy. Mostly the TotalSegmentator names follow this standard.
For some classes they differ which you can see in the table below.

[Here](totalsegmentator/resources/totalsegmentator_snomed_mapping.csv) you can find a mapping of the TotalSegmentator classes to SNOMED-CT codes.

| Index | TotalSegmentator name            | TA2 name                    |
| ----: | -------------------------------- | --------------------------- |
|     1 | spleen                           |                             |
|     2 | kidney\_right                    |                             |
|     3 | kidney\_left                     |                             |
|     4 | gallbladder                      |                             |
|     5 | liver                            |                             |
|     6 | stomach                          |                             |
|     7 | pancreas                         |                             |
|     8 | adrenal\_gland\_right            | suprarenal gland            |
|     9 | adrenal\_gland\_left             | suprarenal gland            |
|    10 | lung\_upper\_lobe\_left          | superior lobe of left lung  |
|    11 | lung\_lower\_lobe\_left          | inferior lobe of left lung  |
|    12 | lung\_upper\_lobe\_right         | superior lobe of right lung |
|    13 | lung\_middle\_lobe\_right        | middle lobe of right lung   |
|    14 | lung\_lower\_lobe\_right         | inferior lobe of right lung |
|    15 | esophagus                        |                             |
|    16 | trachea                          |                             |
|    17 | thyroid\_gland                   |                             |
|    18 | small\_bowel                     | small intestine             |
|    19 | duodenum                         |                             |
|    20 | colon                            |                             |
|    21 | urinary\_bladder                 |                             |
|    22 | prostate                         |                             |
|    23 | kidney\_cyst\_left               |                             |
|    24 | kidney\_cyst\_right              |                             |
|    25 | sacrum                           |                             |
|    26 | vertebrae\_S1                    |                             |
|    27 | vertebrae\_L5                    |                             |
|    28 | vertebrae\_L4                    |                             |
|    29 | vertebrae\_L3                    |                             |
|    30 | vertebrae\_L2                    |                             |
|    31 | vertebrae\_L1                    |                             |
|    32 | vertebrae\_T12                   |                             |
|    33 | vertebrae\_T11                   |                             |
|    34 | vertebrae\_T10                   |                             |
|    35 | vertebrae\_T9                    |                             |
|    36 | vertebrae\_T8                    |                             |
|    37 | vertebrae\_T7                    |                             |
|    38 | vertebrae\_T6                    |                             |
|    39 | vertebrae\_T5                    |                             |
|    40 | vertebrae\_T4                    |                             |
|    41 | vertebrae\_T3                    |                             |
|    42 | vertebrae\_T2                    |                             |
|    43 | vertebrae\_T1                    |                             |
|    44 | vertebrae\_C7                    |                             |
|    45 | vertebrae\_C6                    |                             |
|    46 | vertebrae\_C5                    |                             |
|    47 | vertebrae\_C4                    |                             |
|    48 | vertebrae\_C3                    |                             |
|    49 | vertebrae\_C2                    |                             |
|    50 | vertebrae\_C1                    |                             |
|    51 | heart                            |                             |
|    52 | aorta                            |                             |
|    53 | pulmonary\_vein                  |                             |
|    54 | brachiocephalic\_trunk           |                             |
|    55 | subclavian\_artery\_right        |                             |
|    56 | subclavian\_artery\_left         |                             |
|    57 | common\_carotid\_artery\_right   |                             |
|    58 | common\_carotid\_artery\_left    |                             |
|    59 | brachiocephalic\_vein\_left      |                             |
|    60 | brachiocephalic\_vein\_right     |                             |
|    61 | atrial\_appendage\_left          |                             |
|    62 | superior\_vena\_cava             |                             |
|    63 | inferior\_vena\_cava             |                             |
|    64 | portal\_vein\_and\_splenic\_vein | hepatic portal vein         |
|    65 | iliac\_artery\_left              | common iliac artery         |
|    66 | iliac\_artery\_right             | common iliac artery         |
|    67 | iliac\_vena\_left                | common iliac vein           |
|    68 | iliac\_vena\_right               | common iliac vein           |
|    69 | humerus\_left                    |                             |
|    70 | humerus\_right                   |                             |
|    71 | scapula\_left                    |                             |
|    72 | scapula\_right                   |                             |
|    73 | clavicula\_left                  | clavicle                    |
|    74 | clavicula\_right                 | clavicle                    |
|    75 | femur\_left                      |                             |
|    76 | femur\_right                     |                             |
|    77 | hip\_left                        |                             |
|    78 | hip\_right                       |                             |
|    79 | spinal\_cord                     |                             |
|    80 | gluteus\_maximus\_left           | gluteus maximus muscle      |
|    81 | gluteus\_maximus\_right          | gluteus maximus muscle      |
|    82 | gluteus\_medius\_left            | gluteus medius muscle       |
|    83 | gluteus\_medius\_right           | gluteus medius muscle       |
|    84 | gluteus\_minimus\_left           | gluteus minimus muscle      |
|    85 | gluteus\_minimus\_right          | gluteus minimus muscle      |
|    86 | autochthon\_left                 |                             |
|    87 | autochthon\_right                |                             |
|    88 | iliopsoas\_left                  | iliopsoas muscle            |
|    89 | iliopsoas\_right                 | iliopsoas muscle            |
|    90 | brain                            |                             |
|    91 | skull                            |                             |
|    92 | rib\_left\_1                     |                             |
|    93 | rib\_left\_2                     |                             |
|    94 | rib\_left\_3                     |                             |
|    95 | rib\_left\_4                     |                             |
|    96 | rib\_left\_5                     |                             |
|    97 | rib\_left\_6                     |                             |
|    98 | rib\_left\_7                     |                             |
|    99 | rib\_left\_8                     |                             |
|   100 | rib\_left\_9                     |                             |
|   101 | rib\_left\_10                    |                             |
|   102 | rib\_left\_11                    |                             |
|   103 | rib\_left\_12                    |                             |
|   104 | rib\_right\_1                    |                             |
|   105 | rib\_right\_2                    |                             |
|   106 | rib\_right\_3                    |                             |
|   107 | rib\_right\_4                    |                             |
|   108 | rib\_right\_5                    |                             |
|   109 | rib\_right\_6                    |                             |
|   110 | rib\_right\_7                    |                             |
|   111 | rib\_right\_8                    |                             |
|   112 | rib\_right\_9                    |                             |
|   113 | rib\_right\_10                   |                             |
|   114 | rib\_right\_11                   |                             |
|   115 | rib\_right\_12                   |                             |
|   116 | sternum                          |                             |
|   117 | costal\_cartilages               |                             |

**Class map for task `total_mr`:**


| Index | TotalSegmentator name            | TA2 name               |
| ----: | -------------------------------- | ---------------------- |
|     1 | spleen                           |                        |
|     2 | kidney\_right                    |                        |
|     3 | kidney\_left                     |                        |
|     4 | gallbladder                      |                        |
|     5 | liver                            |                        |
|     6 | stomach                          |                        |
|     7 | pancreas                         |                        |
|     8 | adrenal\_gland\_right            | suprarenal gland       |
|     9 | adrenal\_gland\_left             | suprarenal gland       |
|    10 | lung\_left                       |                        |
|    11 | lung\_right                      |                        |
|    12 | esophagus                        |                        |
|    13 | small\_bowel                     | small intestine        |
|    14 | duodenum                         |                        |
|    15 | colon                            |                        |
|    16 | urinary\_bladder                 |                        |
|    17 | prostate                         |                        |
|    18 | sacrum                           |                        |
|    19 | vertebrae                        |                        |
|    20 | intervertebral\_discs            |                        |
|    21 | spinal\_cord                     |                        |
|    22 | heart                            |                        |
|    23 | aorta                            |                        |
|    24 | inferior\_vena\_cava             |                        |
|    25 | portal\_vein\_and\_splenic\_vein | hepatic portal vein    |
|    26 | iliac\_artery\_left              | common iliac artery    |
|    27 | iliac\_artery\_right             | common iliac artery    |
|    28 | iliac\_vena\_left                | common iliac vein      |
|    29 | iliac\_vena\_right               | common iliac vein      |
|    30 | humerus\_left                    |                        |
|    31 | humerus\_right                   |                        |
|    32 | scapula\_left                    |                        |
|    33 | scapula\_right                   |                        |
|    34 | clavicula\_left                  |                        |
|    35 | clavicula\_right                 |                        |
|    36 | femur\_left                      |                        |
|    37 | femur\_right                     |                        |
|    38 | hip\_left                        |                        |
|    39 | hip\_right                       |                        |
|    40 | gluteus\_maximus\_left           | gluteus maximus muscle |
|    41 | gluteus\_maximus\_right          | gluteus maximus muscle |
|    42 | gluteus\_medius\_left            | gluteus medius muscle  |
|    43 | gluteus\_medius\_right           | gluteus medius muscle  |
|    44 | gluteus\_minimus\_left           | gluteus minimus muscle |
|    45 | gluteus\_minimus\_right          | gluteus minimus muscle |
|    46 | autochthon\_left                 |                        |
|    47 | autochthon\_right                |                        |
|    48 | iliopsoas\_left                  | iliopsoas muscle       |
|    49 | iliopsoas\_right                 | iliopsoas muscle       |
|    50 | brain                            |                        |
