# Details on how the prediction of the contrast phase is done

TotalSegmentator is used to predict the following structures:
```python
["liver", "pancreas", "urinary_bladder", "gallbladder",
"heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
"iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
"pulmonary_vein", "brain", "colon", "small_bowel",
"internal_carotid_artery_right", "internal_carotid_artery_left",
"internal_jugular_vein_right", "internal_jugular_vein_left"]
```
Then the median intensity (HU value) of each structure is used as feature for a xgboost classifier
to predict the post injection time (pi_time). The pi_time can be mapped to the contrast phase
then. It classifies into `native`, `arterial_early`, `arterial_late`, and `portal_venous` phase.
The classifier was trained on the TotalSegmentator dataset and therefore works with all sorts 
of different CT images.

Results on 5-fold cross validation:

- Mean absolute error (MAE): 5.55s
- F1 scores for each class:
  - native: 0.980
  - arterial_early+late: 0.915
  - portal: 0.940

The results contain a probability for each class which is high if the predicted pi_time is close to the ideal
pi_time for the given phase. Moreover, the classifier is an ensemble of 5 models. The output contains the 
standard deviation of the predictions which can be used as a measure of confidence. If it is low the 5 models
give similar predictions which is a good sign.

