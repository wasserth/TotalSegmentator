# Evans index calculation

The following approach is used to calculate the Evans index:

1. Skull stripping using the TotalSegmentator brain mask
2. Rigid registration of brain to a CT brain atlas (custom average of several images) to make sure images always has the same orientation and the brain axes align with the image axes
3. Segment frontal horn of ventricles using TotalSegmentator and also register to the brain atlas
4. Remove small unconnected segmentation blobs
5. Dilate the brain mask to fill the entire space inside of skull
6. Calculate max diameter along the x-axis (left-right) for brain and frontal horn of ventricles
7. Calculate Evans index
8. Plot on top of skull and ventricles mask


## Notes

* the returned brain mask is the raw brain mask from TotalSegmentator, not the dilated one which fills the entire space inside of skull
