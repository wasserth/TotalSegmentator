# Details on how the prediction of body size, weight, age and sex is done

TotalSegmentator is used to predict the following structures:
```python
["..."]
```
Then the volume and median intensity (HU value) of each structure is used as feature for a xgboost classifier.

## Weight prediction

### CT
Number of training images: 46972    (images withg abdomen OR thorax at least partially visible)
Mean absolute error (MAE): 3.4 kg   (evaluated on 16181 images with abdomen AND thorax at least partially visible)

### MR
Number of training images: 31901    (images withg abdomen OR thorax at least partially visible)
Mean absolute error (MAE): 5.5 kg   (evaluated on 1145 images with abdomen AND thorax at least partially visible)


## Size prediction

### CT
Number of training images: 46972    (images withg abdomen OR thorax at least partially visible)
Mean absolute error (MAE): 3.8 cm   (evaluated on 16181 images with abdomen AND thorax at least partially visible)

### MR
Number of training images: 31901    (images withg abdomen OR thorax at least partially visible)
Mean absolute error (MAE): 4.0 cm   (evaluated on 1145 images with abdomen AND thorax at least partially visible)


## Age prediction

### CT
Number of training images: 46972    (images withg abdomen OR thorax at least partially visible)
Mean absolute error (MAE): 4.9 years   (evaluated on 16181 images with abdomen AND thorax at least partially visible)

### MR
Number of training images: 31901    (images withg abdomen OR thorax at least partially visible)
Mean absolute error (MAE): 5.3 years   (evaluated on 1145 images with abdomen AND thorax at least partially visible)


## Sex prediction

### CT
Number of training images: 46972    (images withg abdomen OR thorax at least partially visible)
F1 score: 0.995   (evaluated on 16181 images with abdomen AND thorax at least partially visible)

### MR
Number of training images: 31901    (images withg abdomen OR thorax at least partially visible)
F1 score: 0.901   (evaluated on 1145 images with abdomen AND thorax at least partially visible)



## Info

The bigger the field of view the better the prediction (e.g. complete abdomen and thorax give a lot better results, than images with only the pelvis visible).

The classifier is an ensemble of 5 models. The output contains the 
standard deviation of the predictions which can be used as a measure of confidence. If it is low the 5 models
give similar predictions which is a good sign.
