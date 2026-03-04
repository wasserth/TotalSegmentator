# Details on how the prediction of body size, weight, age and sex is done

TotalSegmentator is used to predict the following structures:
```python
organs = [
    'gluteus_maximus_left', 'hip_right', 'spinal_cord', 'heart', 'spleen', 'hip_left',
    'clavicula_left', 'scapula_left', 'gluteus_maximus_right', 'gallbladder', 'humerus_right',
    'gluteus_minimus_right', 'autochthon_left', 'gluteus_minimus_left', 'scapula_right',
    'femur_right', 'pancreas', 'prostate', 'aorta', 'liver', 'iliopsoas_left',
    'clavicula_right', 'brain', 'gluteus_medius_left', 'humerus_left', 'gluteus_medius_right',
    'kidney_left', 'femur_left', 'kidney_right', 'autochthon_right', 'iliopsoas_right',
    'lung_left', 'lung_right'
]

vertebrae = [
    'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 'vertebrae_C5',
    'vertebrae_C6', 'vertebrae_C7', 'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3',
    'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8',
    'vertebrae_T9', 'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12', 'vertebrae_L1',
    'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5'
]

tissue_types = ['subcutaneous_fat', 'torso_fat', 'skeletal_muscle']
```
For CT, the 5 lung lobes are combined into `lung_left` and `lung_right`. Additionally, for each tissue type a slice is extracted at each vertebra level (tissue_type × vertebra combinations).

Then the volume and median intensity (HU value) of each structure is used as feature for a xgboost classifier.

## Weight prediction

### CT
Number of training images: 46972    (images with abdomen OR thorax at least partially visible)  
MAE (mean absolute error) on CV (cross-validation) : 3.4 kg   (evaluated on 16181 images with abdomen AND thorax at least partially visible)  
MAE on test set: 3.66 kg (stddev 4.91 kg)   (hold-out test set of 501 CT images)

### MR
Number of training images: 31901    (images with abdomen OR thorax at least partially visible)  
MAE on CV: 5.5 kg   (evaluated on 1145 images with abdomen AND thorax at least partially visible)  


## Size prediction

### CT
Number of training images: 46972    (images with abdomen OR thorax at least partially visible)  
MAE on CV: 3.8 cm   (evaluated on 16181 images with abdomen AND thorax at least partially visible)  
MAE on test set: 3.79 cm (stddev 3.34 cm)   (hold-out test set of 501 CT images)

### MR
Number of training images: 31901    (images with abdomen OR thorax at least partially visible)  
Mean absolute error (MAE): 4.0 cm   (evaluated on 1145 images with abdomen AND thorax at least partially visible)  


## Age prediction

### CT
Number of training images: 46972    (images with abdomen OR thorax at least partially visible)  
MAE on CV: 4.9 years   (evaluated on 16181 images with abdomen AND thorax at least partially visible)  
MAE on test set: 5.56 years (stddev 5.36 years)   (hold-out test set of 501 CT images)

### MR
Number of training images: 31901    (images with abdomen OR thorax at least partially visible)  
Mean absolute error (MAE): 5.3 years   (evaluated on 1145 images with abdomen AND thorax at least partially visible)  


## Sex prediction

### CT
Number of training images: 46972    (images with abdomen OR thorax at least partially visible)  
F1 score on CV: 0.995   (evaluated on 16181 images with abdomen AND thorax at least partially visible)  
Accuracy on test set: 0.972   (hold-out test set of 501 CT images)

### MR
Number of training images: 31901    (images with abdomen OR thorax at least partially visible)  
F1 score: 0.901   (evaluated on 1145 images with abdomen AND thorax at least partially visible)  


## Info

**Do not use for age < 16 years, since the model was not trained on children.**

**The bigger the field of view the better the prediction (e.g. complete abdomen and thorax give a lot better results, than images with only the pelvis visible).**

The classifier is an ensemble of 5 models. The output contains the 
standard deviation of the predictions which can be used as a measure of confidence. If it is low the 5 models
give similar predictions which is a good sign.

The following plots show the distribution of the training data. If you try to predict cases out of this distribution, the model will likely not perform well.

![Alt text](resources/imgs/body_stats_train_distr_PatientWeight.png)

![Alt text](resources/imgs/body_stats_train_distr_PatientSize.png)

![Alt text](resources/imgs/body_stats_train_distr_PatientAge.png)

![Alt text](resources/imgs/body_stats_train_distr_PatientSex.png)


## Limitations

The model was trained on clinical data. This makes the model more robust and more generalizable to other clinical settings (e.g. in contrast to a model trained on some population study like UK Biobank). However, sometimes the body weight and size are not exactly measured but only estimated when being added to the DICOM header by clinicians. This reduces the accuracy of the model.


## License of the body stats model

CC-BY-NC 4.0

