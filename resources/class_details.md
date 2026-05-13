# Details about `total` structures

* If a subject has 13 ribs (instead of default 12), then the 13th rib is labeled as `rib_left/right_12` (so rib 12 and 13 get the same label). 
* In case of pleural the effusion the lung lobe segmentations often contain part of the pleural effusion. If you do not want this, run task `pleural_pericard_effusion` and subtract the resulting pleural effusion segmentation from the lung lobe segmentations.
* `colon` and `small_bowel` are sometimes mixed up by the model.