import sys
import os
import itertools
import random
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_map
from fury import window, actor, ui, io, utils

from totalsegmentator.vtk_utils import contour_from_roi_smooth, plot_mask


random.seed(1234)
np.random.seed(1234)

random_colors = np.random.rand(100, 4)

roi_groups = [
    ['humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left',
     'clavicula_right', 'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum',
     'colon', 'heart_trachea'],
    ['organ_spleen', 'organ_right_kidney', 'organ_left_kidney', 'organ_gallbladder',
     'organ_right_adrenal_gland', 'organ_left_adrenal_gland', 
     'gluteus_medius_left', 'gluteus_medius_right',
     'dheart_left_atrium', 'dheart_right_atrium', 'dheart_myocardium'
     ],
    ['iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right',
     'organ_aorta', 'organ_inferior_vena_cava', 
     'organ_portal_vein_and_splenic_vein', 'heart_esophagus'],
    ['small_bowel', 'organ_stomach', 'lung_left_upper_lobe', 'lung_right_upper_lobe', 'face'],
    ['lung_left_lower_lobe', 'lung_right_middle_lobe', 'lung_right_lower_lobe',
     'organ_pancreas', 'brain'],
    ['vertebrae_L6', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2',
     'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9',
     'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4',
     'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6',
     'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1',
     'gluteus_maximus_left', 'gluteus_maximus_right'],
    ['rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6',
     'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12',
     'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6',
     'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 
     'rib_right_12', 'urinary_bladder', 'duodenum',
     'gluteus_minimus_left', 'gluteus_minimus_right'],
    ['organ_liver', 'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right',
     'dheart_left_ventricle', 'dheart_right_ventricle', 'dheart_pulmunary_artery']
]


def plot_roi_group(subject_path, scene, rois, x, y, roi_dir="roi"):
    ref_img = nib.load(subject_path)
    roi_actors = []
    for idx, roi in enumerate(rois):
        color = random_colors[idx]
        img_path = roi_dir / f"{roi}.nii.gz"
        if img_path.exists():
            img = nib.load(img_path)
            affine = img.affine
            data = img.get_fdata()
        else:
            affine = ref_img.affine
            data = np.zeros(ref_img.shape)

        opacity = 1
        # opacity = 0.5  # Not good. Not so easy to see failures anymore.

        smoothing=20
        # smoothing=0  # faster

        if data.max() > 0:  # empty mask
            affine[:3, 3] = 0  # make offset the same for all subjects
            cont_actor = plot_mask(scene, data, affine, x, y, smoothing=smoothing,
                                color=color, opacity=opacity)
            scene.add(cont_actor)
            roi_actors.append(cont_actor)


def plot_subject(subject_path, output_path, df=None, roi_dir="roi"):
    subject_width = 330
    # subject_height = 700
    nr_cols = 10

    window_size = (1800, 400)

    scene = window.Scene()
    showm = window.ShowManager(scene, size=window_size, reset_camera=False)
    showm.initialize()

    ct_img = nib.load(subject_path)
    data = ct_img.get_fdata()
    data = data.transpose(1, 2, 0)  # Show sagittal view
    data = data[::-1, :, :]
    value_range = (-115, 225)  # soft tissue window
    slice_actor = actor.slicer(data, ct_img.affine, value_range)
    slice_actor.SetPosition(0, 0, 0)
    scene.add(slice_actor)

    # Plot 3D rois
    for idx, roi_group in enumerate(roi_groups):
        idx += 1  # increase by 1 because 0 is the ct image
        x = (idx % nr_cols) * subject_width
        # y = (idx // nr_cols) * subject_height
        y = 0
        plot_roi_group(subject_path, scene, roi_group, x, y, roi_dir)

    # window.show(scene, size=(900, 700), reset_camera=False)
    # print(scene.get_camera())

    # This needs to be adapted when changing number of subjects I display
    # important: have to set reset_camera=False for this to work
    # scene.set_camera(position=(612., 331., 1782.),  # decrease z: zoom a bit closer
    #                  focal_point=(612., 331., 228.),
    #                  view_up=(0.0, 1.0, 0.0))
    
    scene.projection(proj_type='parallel')
    scene.reset_camera_tight(margin_factor=1.02)  # need to do reset_camera=False in record for this to work in

    window.record(scene, size=window_size,
                    out_path=output_path, reset_camera=False)  # , reset_camera=False
    scene.clear()


def generate_preview(ct_in, file_out, roi_dir):
    from xvfbwrapper import Xvfb
    vdisplay = Xvfb()
    vdisplay.start()
    try:
        plot_subject(ct_in, file_out, df=None, roi_dir=roi_dir)
    finally:
        vdisplay.stop()
