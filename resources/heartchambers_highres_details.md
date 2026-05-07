# Details on the heartchambers_highres task

Number of training images: 1559  (1502 training, 57 validation)

The training is made up of paired images of contrast and non-contrast scans. These two scans were acquired in the same session and non-linear registration was used to register the non-contrast image to the contrast image for even better alignment. The annotation was performed on the contrast images and then the segmentations were transferred to the non-contrast image. During the training the model was provided with both contrast and non-contrast images.

The model works on a resolution of 0.7265625 x 0.72265625 x 1.0 mm.

Dice across all classes on the validation set: 0.950
