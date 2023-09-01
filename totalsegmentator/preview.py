import sys
import os
import itertools
import pickle
from pathlib import Path
from pprint import pprint
import gc

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from fury import window, actor, ui, io, utils

from totalsegmentator.vtk_utils import contour_from_roi_smooth, plot_mask
from totalsegmentator.map_to_binary import class_map


np.random.seed(1234)
random_colors = np.random.rand(100, 4)

roi_groups = {
    "total": [
        ["humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left",
         "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right", "sacrum",         
        #  "patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet", "ulna", "radius", "carpal", "metacarpal", "phalanges_hand",
         "colon", "trachea", "skull"],
        ["spleen", "kidney_right", "kidney_left", "gallbladder",
         "adrenal_gland_right", "adrenal_gland_left",
         "gluteus_medius_left", "gluteus_medius_right",
         "heart",
        #  "heart_atrium_left", "heart_atrium_right", "heart_myocardium",
         "kidney_cyst_left", "kidney_cyst_right", "spinal_cord", "prostate", "thyroid_gland"],
        ["iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right",
         "aorta", "inferior_vena_cava",
         "portal_vein_and_splenic_vein", "esophagus",
         "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left", 
         "common_carotid_artery_right", "common_carotid_artery_left",  
         "atrial_appendage_left"],
        ["small_bowel", "stomach", "lung_upper_lobe_left",
         "lung_upper_lobe_right"],
        ["lung_lower_lobe_left", "lung_middle_lobe_right", "lung_lower_lobe_right",
         "pancreas", "brain"],
        ["vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2",
         "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9",
         "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4",
         "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6",
         "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
         "gluteus_maximus_left", "gluteus_maximus_right"],
        ["rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
         "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
         "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
         "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11",
         "rib_right_12", "urinary_bladder", "duodenum",
         "gluteus_minimus_left", "gluteus_minimus_right", "sternum", "costal_cartilages"],
        ["liver", "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right",
        #  "heart_ventricle_left", "heart_ventricle_right", "pulmonary_artery", 
        "pulmonary_vein",
         "superior_vena_cava", "brachiocephalic_vein_left", "brachiocephalic_vein_right"]
    ],
    "lung_vessels": [
        ["lung_trachea_bronchia"],
        ["lung_vessels"]
    ],
    "covid": [
        ["lung_covid_infiltrate"]
    ],
    "cerebral_bleed": [
        ["intracerebral_hemorrhage"]
    ],
    "hip_implant": [
        ["hip_implant"]
    ],
    "coronary_arteries": [
        ["coronary_arteries"]
    ],
    "body": [
        ["body_trunc", "body_extremities"]
    ],
    "pleural_pericard_effusion": [
        ["pleural_effusion", "pericardial_effusion"]
    ],
    "liver_vessels": [
        ["liver_vessels", "liver_tumor"]
    ],
    "vertebrae_body": [
        ["vertebrae_body"]
    ],
    "heartchambers_highres": [
        ["heart_myocardium"],
        ["heart_atrium_left", "heart_ventricle_left"],
        ["heart_atrium_right", "heart_ventricle_right"],
        ["aorta", "pulmonary_artery"]
    ],
    "appendicular_bones": [
        ["patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet", 
         "ulna", "radius", "carpal", "metacarpal", "phalanges_hand"]
    ],
    "tissue_types": [
        ["subcutaneous_fat"],
        ["torso_fat"],
        ["skeletal_muscle"]
    ],
    "face": [
        ["face"]
    ],
    # "aortic_branches_test": [
    #     ["brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left", "aorta",
    #     "common_carotid_artery_right", "common_carotid_artery_left"],
    #     ["superior_vena_cava", 
    #     "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left"],
    #     ["pulmonary_vein", "pulmonary_artery"],
    #     ["heart_atrium_left", "heart_atrium_right", "thyroid_gland"]
    # ],
    "test": [
        ["ulna"]
    ]
}


def plot_roi_group(ref_img, scene, rois, x, y, smoothing, roi_data, affine, task_name):
    # ref_img = nib.load(subject_path)
    # roi_actors = []

    for idx, roi in enumerate(rois):
        color = random_colors[idx]

        classname_2_idx = {v: k for k, v in class_map[task_name].items()}
        data = roi_data == classname_2_idx[roi]
        # data = data.astype(np.uint8)  # needed?

        if data.max() > 0:  # empty mask
            affine[:3, 3] = 0  # make offset the same for all subjects
            cont_actor = plot_mask(scene, data, affine, x, y, smoothing=smoothing,
                                color=color, opacity=1)
            scene.add(cont_actor)
            # roi_actors.append(cont_actor)


def plot_subject(ct_img, output_path, df=None, roi_data=None, smoothing=20,
                 task_name="total"):
    subject_width = 330
    # subject_height = 700
    nr_cols = 10

    window_size = (1800, 400)
    # window_size = (1800, 1200)  # if we need higher res image of single class

    scene = window.Scene()
    showm = window.ShowManager(scene, size=window_size, reset_camera=False)
    showm.initialize()

    # ct_img = nib.load(subject_path)
    data = ct_img.get_fdata()
    data = data.transpose(1, 2, 0)  # Show sagittal view
    data = data[::-1, :, :]
    value_range = (-115, 225)  # soft tissue window
    slice_actor = actor.slicer(data, ct_img.affine, value_range)
    slice_actor.SetPosition(0, 0, 0)
    scene.add(slice_actor)

    # Plot 3D rois
    for idx, roi_group in enumerate(roi_groups[task_name]):
        idx += 1  # increase by 1 because 0 is the ct image
        x = (idx % nr_cols) * subject_width
        # y = (idx // nr_cols) * subject_height
        y = 0
        plot_roi_group(ct_img, scene, roi_group, x, y, smoothing, roi_data, ct_img.affine,
                       task_name)

    # window.show(scene, size=(900, 700), reset_camera=False)
    # print(scene.get_camera())

    # This needs to be adapted when changing number of subjects I display
    # important: have to set reset_camera=False for this to work
    # scene.set_camera(position=(612., 331., 1782.),  # decrease z: zoom a bit closer
    #                  focal_point=(612., 331., 228.),
    #                  view_up=(0.0, 1.0, 0.0))
    
    scene.projection(proj_type="parallel")
    scene.reset_camera_tight(margin_factor=1.02)  # need to do reset_camera=False in record for this to work in

    output_path.parent.mkdir(parents=True, exist_ok=True)
    window.record(scene, size=window_size,
                  out_path=output_path, reset_camera=False)  # , reset_camera=False
    scene.clear()


def generate_preview(ct_in, file_out, roi_data, smoothing, task_name):
    from xvfbwrapper import Xvfb
    # do not set random seed, otherwise can not call xvfb in parallel, because all generate same tmp dir
    with Xvfb() as xvfb:
        plot_subject(ct_in, file_out, None, roi_data, smoothing, task_name)
