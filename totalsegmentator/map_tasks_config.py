# Task standard default parameters
DEFAULT_CONFIG = {
    "resample": None,
    "crop": None,
    "crop_model": None,
    "crop_addon": [3, 3, 3],  # default value
    "robust_crop": False,
    "cascade": None,
    "remove_outside": None,
    "remove_outside_dilation": None,
    "remove_mask": None,
    "modality": None,
    "plans": None,
    "model": "3d_fullres",
    "folds": [0],
    "disallow_fast": False,
    "commercial": False,
}

# Important: 'resample' expects [x,y,z] but in nnUNet plans.json file it is [z,y,x]. So when copying from plans.json make sure to reverse the order.

# Task custom parameters
TASK_CONFIGS = {
    # Tasks with sub-modes (fast / fastest / default)
    "total": {
        "sub_modes": {
            "fast": {
                "task_id": 297,
                "resample": 3.0,
                "trainer": "nnUNetTrainer_4000epochs_NoMirroring",
                # "trainer": "nnUNetTrainerNoMirroring",
            },
            "fastest": {
                "task_id": 298,
                "resample": 6.0,
                "trainer": "nnUNetTrainer_4000epochs_NoMirroring",
            },
            "default": {
                "task_id": [291, 292, 293, 294, 295],
                "resample": 1.5,
                "trainer": "nnUNetTrainerNoMirroring",
            },
        }
    },
    "total_v3": {
        "sub_modes": {
            "fast": {
                "task_id": 836,
                "resample": 3.0,
                "trainer": "nnUNetTrainer_4000epochs_NoMirroring",
            },
            "fastest": {
                "task_id": 837,
                "resample": 6.0,
                "trainer": "nnUNetTrainer_4000epochs_NoMirroring",
            },
            "default": {
                "task_id": [831, 832, 833, 834, 835],
                "resample": 1.5,
                "trainer": "nnUNetTrainerNoMirroring",
            },
        }
    },
    "total_mr": {
        "sub_modes": {
            "fast": {
                "task_id": 852,
                "resample": 3.0,
                "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
                # "trainer": "nnUNetTrainerNoMirroring",
            },
            "fastest": {
                "task_id": 853,
                "resample": 6.0,
                "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
            },
            "default": {
                "task_id": [850, 851],
                "resample": 1.5,
                "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
            },
        }
    },
    "body": {
        "sub_modes": {
            "fast": {"task_id": 300, "resample": 6.0, "trainer": "nnUNetTrainer"},
            "default": {"task_id": 299, "resample": 1.5, "trainer": "nnUNetTrainer"},
        }
    },
    "body_mr": {
        "sub_modes": {
            "fast": {
                "task_id": 598,  # todo: train
                "resample": 6.0,
                "trainer": "nnUNetTrainer_DASegOrd0",
            },
            "default": {
                "task_id": 597,
                "resample": 1.5,
                "trainer": "nnUNetTrainer_DASegOrd0",
            },
        }
    },

    # Direct task mappings
    # todo: add to download and preview
    # "total_highres_test": {
    #     # task_id = 955
    #     "task_id": 956,
    #     # resample = [0.75, 0.75, 1.0]
    #     "resample": [0.78125, 0.78125, 1.0],
    #     "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
    #     "crop_addon": [30, 30, 30],
    #     "crop": ["liver", "spleen", "colon", "small_bowel", "stomach", "lung_upper_lobe_left", "lung_upper_lobe_right", "aorta"], # abdomen_thorax
    #     # model = "3d_fullres_high"
    #     # model = "3d_fullres_high_bigPS"
    #     "model": "3d_fullres",
    #     "cascade": True,
    # },
    "total_highres_test": {
        "task_id": 957,
        "resample": [0.75, 0.75, 1.0],
        "trainer": "nnUNetTrainerNoMirroring",
        # "crop_addon": [30, 30, 30],
        # "crop": ["liver", "spleen", "colon", "small_bowel", "stomach", "lung_upper_lobe_left", "lung_upper_lobe_right", "aorta"], # abdomen_thorax
        "model": "3d_fullres_high",
        # "model": "3d_fullres_high_bigPS",
        "cascade": False,
    },
    "lung_vessels": {
        "task_id": 117,
        "resample": [0.703125, 0.703125, 1.0],
        "trainer": "nnUNetTrainerSkeletonRecall",
        "crop": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "robust_crop": True,
        # todo: adapt code to support removing outside of body (so far only "total" task classes supported)
        # "remove_outside": ["body_trunc"],
        # "remove_outside_dilation": 0,  # mm
    },
    "lung_vessels_LEGACY": {
        "task_id": 258,
        "trainer": "nnUNetTrainer",
        "crop": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "disallow_fast": True,
    },
    # "covid": {
    #     "task_id": 201,
    #     "resample": None,
    #     "trainer": "nnUNetTrainer",
    #     "crop": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    #              "lung_middle_lobe_right", "lung_lower_lobe_right"],
    #     "model": "3d_fullres",
    #     "folds": [0],
    #     # print("WARNING: The COVID model finds many types of lung opacity not only COVID. Use with care!")
    #     "disallow_fast": True,
    # },
    "cerebral_bleed": {
        "task_id": 150,
        "trainer": "nnUNetTrainer",
        "crop": ["brain"],
        "disallow_fast": True,
    },
    "hip_implant": {
        "task_id": 260,
        "trainer": "nnUNetTrainer",
        "crop": ["femur_left", "femur_right", "hip_left", "hip_right"],
        "disallow_fast": True,
    },
    "vertebrae_mr": {
        "task_id": 756,
        "resample": 1.5,
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        # "disallow_fast": True # probably should have but does not for some reason.
    },
    "pleural_pericard_effusion": {
        "task_id": 315,
        "trainer": "nnUNetTrainer",
        "crop": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "crop_addon": [50, 50, 50],
        "folds": None,
        "disallow_fast": True,
    },
    "liver_vessels": {
        "task_id": 8,
        "trainer": "nnUNetTrainer",
        "crop": ["liver"],
        "crop_addon": [20, 20, 20],
        "disallow_fast": True,
    },
    "head_glands_cavities": {
        "task_id": 775,
        "resample": [0.75, 0.75, 1.0],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["skull"],
        "crop_addon": [10, 10, 10],
        "model": "3d_fullres_high",
        "disallow_fast": True,
    },
    "headneck_bones_vessels": {
        "task_id": 776,
        "resample": [0.75, 0.75, 1.0],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        # "crop": ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"],
        # "crop_addon": [10, 10, 10],
        "crop": ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"],
        "crop_addon": [40, 40, 40],
        "model": "3d_fullres_high",
        "disallow_fast": True,
    },
    "head_muscles": {
        "task_id": 777,
        "resample": [0.75, 0.75, 1.0],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["skull"],
        "crop_addon": [10, 10, 10],
        "model": "3d_fullres_high",
        "disallow_fast": True,
    },
    "headneck_muscles": {
        "task_id": [778, 779],
        "resample": [0.75, 0.75, 1.0],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        # "crop": ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"],
        # "crop_addon": [10, 10, 10],
        "crop": ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"],
        "crop_addon": [40, 40, 40],
        "model": "3d_fullres_high",
        "disallow_fast": True,
    },
    "oculomotor_muscles": {
        "task_id": 351,
        "resample": [0.47251562774181366, 0.47251562774181366, 0.8500002026557922],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["skull"],
        "crop_addon": [20, 20, 20],
        "disallow_fast": True,
    },
    "lung_nodules": {
        "task_id": 913,
        "resample": [1.5, 1.5, 1.5],
        "trainer": "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring",
        "crop": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "crop_addon": [10, 10, 10],
        "disallow_fast": True,
    },
    "kidney_cysts": {
        "task_id": 789,
        "resample": [1.5, 1.5, 1.5],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["kidney_left", "kidney_right", "liver", "spleen", "colon"],
        "crop_addon": [10, 10, 10],
        "disallow_fast": True,
    },
    "breasts": {
        "task_id": 527,
        "resample": [1.5, 1.5, 1.5],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "disallow_fast": True,
    },
    "ventricle_parts": {
        "task_id": 552,
        "resample": [0.4384765625, 0.4345703125, 1.0],
        "trainer": "nnUNetTrainerNoMirroring",
        "crop": ["brain"],
        "crop_addon": [0, 0, 0],
        "disallow_fast": True,
    },
    "liver_segments": {
        "task_id": 570,
        "resample": [0.8046879768371582, 0.8046879768371582, 1.5],
        "trainer": "nnUNetTrainerNoMirroring",
        "crop": ["liver"],
        "crop_addon": [10, 10, 10],
        "disallow_fast": True,
    },
    "liver_segments_mr": {
        "task_id": 576,
        "resample": [1.1250001788139343, 1.1875, 3.0],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["liver"],
        "crop_addon": [10, 10, 10],
        "disallow_fast": True,
    },
    "liver_lesions": {
        "task_id": 591,
        "resample": [0.75, 0.75, 1.0],
        "trainer": "nnUNetTrainer",
        "crop": ["liver"],
        "crop_addon": [10, 10, 10],
        "robust_crop": True,
        "model": "3d_fullres_high",
        "disallow_fast": True,
    },
    "liver_lesions_mr": {
        "task_id": 589,
        "resample": [0.8603515625, 0.857421875, 1.0],
        "trainer": "nnUNetTrainer_DASegOrd0",
        "crop": ["liver"],
        "robust_crop": True,
        "disallow_fast": True,
    },
    "craniofacial_structures": {
        "task_id": 115,
        "resample": [0.5, 0.5, 0.5],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["skull"],
        "crop_addon": [20, 20, 20],
        "disallow_fast": True,
    },
    # This model only segments within T4-L4. In training only labels in this region were annotated. Therefore,
    # I do not have to crop to this region, but model automatically only predicts in this region.
    "abdominal_muscles": {
        "task_id": 952,
        "resample": [0.75, 0.75, 1],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["body_trunc"],
        "crop_addon": [5, 5, 5],
        "model": "3d_fullres_high",
        "disallow_fast": True,
    },
    "teeth": {
        "task_id": 113,  # trained with nnUNet v2.6.2; might not be compatible with older nnunet versions
        "resample": [0.5, 0.5, 0.5],
        "trainer": "nnUNetTrainer_onlyMirror01",
        # mandible not included, because would increase FOV way too much -> many wrong segmentation in regions outside of teeth
        # (model was trained on pretty narrow FOV because of CBCT data)
        "crop": ["teeth_lower", "teeth_upper"],
        "crop_model": "craniofacial_structures",
        "crop_addon": [10, 10, 10],
        "model": "3d_lowres_high",
        "disallow_fast": True,
    },
    "trunk_cavities": {
        "task_id": 343,
        "resample": [1.5, 1.5, 1.5],
        "trainer": "nnUNetTrainer",
        "disallow_fast": True,
    },
    "brain_aneurysm": {
        "task_id": 615,
        "resample": [0.390625, 0.390625, 0.5000016391277313],
        "trainer": "nnUNetTrainerDiceTopK10Loss_2000epochs",
        "folds": None,
        "info_msg": "INFO: This task only works with TOF MRI images.\n",
        "disallow_fast": True,
    },
    "vertebrae_body": {
        "task_id": 305,
        "resample": 1.5,
        "trainer": "nnUNetTrainer_DASegOrd0",
        "disallow_fast": True,
    },
    "vertebrae_pp": {
        "task_id": 803,
        "resample": 1.5,
        "trainer": "nnUNetTrainerNoMirroring",
        "disallow_fast": True,
    },
    "vertebrae_pp_refined": {
        "task_id": 803,
        "resample": 1.5,
        "trainer": "nnUNetTrainerNoMirroring",
        "disallow_fast": True,
    },

    # Commercial models
    "heartchambers_highres": {
        "task_id": 301,
        "trainer": "nnUNetTrainer",
        "crop": ["heart"],
        "crop_addon": [5, 5, 5],
        "remove_outside": ["heart", "aorta", "inferior_vena_cava"],
        "remove_outside_dilation": 10,  # mm
        "robust_crop": True,
        "disallow_fast": True,
        "commercial": True,
    },
    "appendicular_bones": {
        "task_id": 304,
        "resample": 1.5,
        "trainer": "nnUNetTrainerNoMirroring",
        "disallow_fast": True,
        "commercial": True,
    },
    "appendicular_bones_mr": {
        "task_id": 855,
        "resample": 1.5,
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "disallow_fast": True,
        "commercial": True,
    },
    "tissue_types": {
        "task_id": 481,
        "resample": 1.5,
        "trainer": "nnUNetTrainer",
        "disallow_fast": True,
        "commercial": True,
    },
    "tissue_types_mr": {
        "task_id": 925,
        "resample": 1.5,
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "disallow_fast": True,
        "commercial": True,
    },
    "tissue_4_types": {
        "task_id": 485,
        "resample": 1.5,
        "trainer": "nnUNetTrainer",
        "disallow_fast": True,
        "commercial": True,
    },
    "face": {
        "task_id": 303,
        "resample": 1.5,
        "trainer": "nnUNetTrainerNoMirroring",
        "disallow_fast": True,
        "commercial": True,
    },
    "face_mr": {
        "task_id": 856,
        "resample": 1.5,
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "disallow_fast": True,
        "commercial": True,
    },
    "brain_structures": {
        "task_id": 409,
        "resample": [0.5, 0.5, 1.0],
        "trainer": "nnUNetTrainer_DASegOrd0",
        "crop": ["brain"],
        "crop_addon": [10, 10, 10],
        "model": "3d_fullres_high",
        "disallow_fast": True,
        "commercial": True,
    },
    "thigh_shoulder_muscles": {
        "task_id": 857,  # at the moment only one mixed model for CT and MR; when annotated all CT samples -> train separate CT model
        "resample": 1.5,
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "disallow_fast": True,
        "commercial": True,
    },
    "thigh_shoulder_muscles_mr": {
        "task_id": 857,
        "resample": 1.5,
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "disallow_fast": True,
        "commercial": True,
    },
    "coronary_arteries": {
        "task_id": 509,
        "resample": [0.7, 0.7, 0.7],
        "trainer": "nnUNetTrainerSkeletonRecall",
        "crop": ["heart"],
        "crop_addon": [20, 20, 20],
        "model": "3d_fullres_high",
        "disallow_fast": True,
        "commercial": True,
    },
    "coronary_arteries_LEGACY": {
        "task_id": 507,
        "resample": [0.7, 0.7, 0.7],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["heart"],
        "crop_addon": [20, 20, 20],
        "model": "3d_fullres_high",
        "disallow_fast": True,
        "commercial": True,
    },
    "aortic_sinuses": {
        "task_id": 920,
        "resample": [0.7, 0.7, 0.7],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        "crop": ["heart"],
        "crop_addon": [0, 0, 0],
        "model": "3d_fullres_high",
        "disallow_fast": True,
        "commercial": True,
    },
    "renal_arteries": {
        "task_id": 710,
        "resample": 1.5,
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        # Model only used for aorta_report and there only working on already cropped image
        "disallow_fast": True,
        "commercial": True,
    },
    "aorta_annulus": {
        "task_id": 713,
        "resample": [0.8, 0.8, 0.8],
        "trainer": "nnUNetTrainer_DASegOrd0",
        # Model only used for aorta_report and there only working on already cropped image
        "model": "3d_fullres_high",
        "folds": [0, 1, 2, 3, 4],
        "disallow_fast": True,
        "commercial": True,
    },
    "aortic_dissection": {
        "task_id": 716,
        "resample": [0.8, 0.8, 0.8],
        "trainer": "nnUNetTrainer_DASegOrd0_NoMirroring",
        # Model only used for aorta_report and there only working on already cropped image
        "model": "3d_fullres_high",
        "folds": [0, 1, 2, 3, 4],
        "disallow_fast": True,
        "commercial": True,
    },
    "pulmonary_artery_landmarks": {
        "task_id": 514,
        "resample": [0.7802734375, 0.7802734375, 1.0],
        "trainer": "nnUNetTrainer",
        "folds": [0, 1, 2, 3, 4],
        "disallow_fast": True,
        "commercial": True,
    },
    "test": {
        "task_id": [517],
        "trainer": "nnUNetTrainerV2",
        "crop": "body",
    },
}

TASK_ID_WEIGHTS_CONFIGS = {
    # Trying to keep original order and comments as much as possible

    # foldername = local and URL foldername
    # version = weights release URL version string
    # rel_path = (optional) extra local path prior to foldername
    # zenodo_weights_url = (optional) previously commented out zenodo url path
    # alt_weights_url = (optional) previously commented out static url path

    # Total CT v2
    291: {
        "foldername": "Dataset291_TotalSegmentator_part1_organs_1559subj",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
    },
    292: {
        "foldername": "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
    },
    293: {
        "foldername": "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
    },
    294: {
        "foldername": "Dataset294_TotalSegmentator_part4_muscles_1559subj",
        "version": "v2.0.0-weights"
        # "zenodo_weights_url": "https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
    },
    295: {
        "foldername": "Dataset295_TotalSegmentator_part5_ribs_1559subj",
        "version": "v2.0.0-weights"
        # "zenodo_weights_url": "https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip",
    },
    297: {
        "foldername": "Dataset297_TotalSegmentator_total_3mm_1559subj",
        "version": "v2.0.0-weights"
        # "zenodo_weights_url": "https://zenodo.org/record/6802052/files/Task256_TotalSegmentator_3mm_1139subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset297_TotalSegmentator_total_3mm_1559subj.zip",

        # "foldername" : "Dataset297_TotalSegmentator_total_3mm_1559subj_v204",
        # "version" : "v2.0.4-weights"

        # "foldername" : "Dataset297_TotalSegmentator_total_3mm_1559subj_v205",
        # "version" : "v2.0.5-weights",
    },
    298: {
        "foldername": "Dataset298_TotalSegmentator_total_6mm_1559subj",
        "version": "v2.0.0-weights",
        # "alt_weights_url": "/static/totalseg_v2/Dataset298_TotalSegmentator_total_6mm_1559subj.zip",

        # "foldername" : "Dataset298_TotalSegmentator_total_6mm_1559subj_v205",
        # "version" : "v2.0.5-weights"
    },

    # Total CT v3
    831: {"foldername": "Dataset831_TotalSegmentator_part1_organs_1830subj", "version": "v3.0.0-weights"},
    832: {"foldername": "Dataset832_TotalSegmentator_part2_vertebrae_1559subj", "version": "v3.0.0-weights"},
    833: {"foldername": "Dataset833_TotalSegmentator_part3_cardiac_1830subj", "version": "v3.0.0-weights"},
    834: {"foldername": "Dataset834_TotalSegmentator_part4_muscles_1830subj", "version": "v3.0.0-weights"},
    835: {"foldername": "Dataset835_TotalSegmentator_part5_ribs_1559subj", "version": "v3.0.0-weights"},
    836: {"foldername": "Dataset836_TotalSegmentator_total_3mm_1559subj", "version": "v3.0.0-weights"},
    837: {"foldername": "Dataset837_TotalSegmentator_total_6mm_1559subj", "version": "v3.0.0-weights"},

    # Body, etc. CT v2
    299: {
        "foldername": "Dataset299_body_1559subj",
        "version": "v2.0.0-weights",
        # "alt_weights_url": "/static/totalseg_v2/Dataset299_body_1559subj.zip",
    },
    300: {
        "foldername": "Dataset300_body_6mm_1559subj",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/7334272/files/Task269_Body_extrem_6mm_1200subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset300_body_6mm_1559subj.zip",
    },
    775: {"foldername": "Dataset775_head_glands_cavities_492subj", "version": "v2.3.0-weights"},
    776: {"foldername": "Dataset776_headneck_bones_vessels_492subj", "version": "v2.3.0-weights"},
    777: {"foldername": "Dataset777_head_muscles_492subj", "version": "v2.3.0-weights"},
    778: {"foldername": "Dataset778_headneck_muscles_part1_492subj", "version": "v2.3.0-weights"},
    779: {"foldername": "Dataset779_headneck_muscles_part2_492subj", "version": "v2.3.0-weights"},
    351: {"foldername": "Dataset351_oculomotor_muscles_18subj", "version": "v2.4.0-weights"},
    789: {"foldername": "Dataset789_kidney_cyst_501subj", "version": "v2.5.0-weights"},
    527: {"foldername": "Dataset527_breasts_1559subj", "version": "v2.5.0-weights"},
    552: {"foldername": "Dataset552_ventricle_parts_38subj", "version": "v2.5.0-weights"},

    955: {"foldername": "Dataset955_TotalSegmentator_highres_part1_organs_110subj", "version": "TODO"},
    956: {"foldername": "Dataset956_TotalSegmentator_highres_part1_organs_cascade_110subj", "version": "TODO"},
    957: {"foldername": "Dataset957_TotalSegmentator_highres_part1_organs_cropBody_127subj", "version": "TODO"},

    # MR models
    850: {"foldername": "Dataset850_TotalSegMRI_part1_organs_1088subj", "version": "v2.5.0-weights"},
    851: {"foldername": "Dataset851_TotalSegMRI_part2_muscles_1088subj", "version": "v2.5.0-weights"},
    852: {"foldername": "Dataset852_TotalSegMRI_total_3mm_1088subj", "version": "v2.5.0-weights"},
    853: {"foldername": "Dataset853_TotalSegMRI_total_6mm_1088subj", "version": "v2.5.0-weights"},
    597: {"foldername": "Dataset597_mri_body_139subj", "version": "v2.5.0-weights"},
    598: {"foldername": "Dataset598_mri_body_6mm_139subj", "version": "v2.5.0-weights"},
    756: {"foldername": "Dataset756_mri_vertebrae_1076subj", "version": "v2.5.0-weights"},

    # Models from other projects
    117: {"foldername": "Dataset117_lung_airways_arteries_veins_282subj", "version": "v2.5.0-weights"},
    258: {
        "foldername": "Dataset258_lung_vessels_248subj",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/7064718/files/Task258_lung_vessels_248subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset258_lung_vessels_248subj.zip",
    },
    200: {"foldername": "Task200_covid_challenge", "version": "TODO"},
    201: {"foldername": "Task201_covid", "version": "TODO"},
    # 152: {"foldername": Path("Task152_icbbig_TN"), "version": "TODO"},
    150: {
        "foldername": "Dataset150_icb_v0",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/7079161/files/Task150_icb_v0.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset150_icb_v0.zip",
    },
    260: {
        "foldername": "Dataset260_hip_implant_71subj",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/7234263/files/Task260_hip_implant_71subj.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset260_hip_implant_71subj.zip",
    },
    315: {
        "foldername": "Dataset315_thoraxCT",
        "version": "v2.0.0-weights",
        # "zenodo_weights_url": "https://zenodo.org/record/7510288/files/Task315_thoraxCT.zip?download=1",
        # "alt_weights_url": "/static/totalseg_v2/Dataset315_thoraxCT.zip",
    },
    8: {"foldername": "Dataset008_HepaticVessel", "version": "v2.4.0-weights"},

    913: {"foldername": "Dataset913_lung_nodules", "version": "v2.5.0-weights"},
    570: {"foldername": "Dataset570_ct_liver_segments", "version": "v2.5.0-weights"},
    576: {"foldername": "Dataset576_mri_liver_segments_120subj", "version": "v2.5.0-weights"},
    591: {"foldername": "Dataset591_ct_liver_lesions_842subj", "version": "v2.5.0-weights"},
    589: {"foldername": "Dataset589_ct_mri_liver_lesions_750subj", "version": "v2.5.0-weights"},
    115: {"foldername": "Dataset115_mandible", "version": "v2.5.0-weights"},
    952: {"foldername": "Dataset952_abdominal_muscles_167subj", "version": "v2.5.0-weights"},
    113: {"foldername": "Dataset113_ToothFairy3", "version": "v2.5.0-weights"},
    343: {"foldername": "Dataset343_mediastinum_1786subj", "version": "v2.5.0-weights"},
    615: {"foldername": "Dataset615_MAXIMUS", "version": "v2.5.0-weights"},
    305: {"foldername": "Dataset305_vertebrae_discs_1559subj", "version": "v2.5.0-weights"},
    803: {"foldername": "Dataset803_TotalSegmentator_vertebrae_inner_1559subj", "version": "v2.5.0-weights"},

    # Commercial models
    304: {"foldername": "Dataset304_appendicular_bones_ext_1559subj"},
    855: {"foldername": "Dataset855_TotalSegMRI_appendicular_bones_1088subj"},
    301: {"foldername": "Dataset301_heart_highres_1559subj"},
    303: {"foldername": "Dataset303_face_1559subj"},
    481: {"foldername": "Dataset481_tissue_1559subj"},
    485: {"foldername": "Dataset485_tissue_4types_1559subj"},
    925: {"foldername": "Dataset925_MRI_tissue_subset_903subj"},
    856: {"foldername": "Dataset856_TotalSegMRI_face_1088subj"},
    409: {"foldername": "Dataset409_neuro_550subj"},
    857: {"foldername": "Dataset857_TotalSegMRI_thigh_shoulder_1088subj"},
    507: {"foldername": "Dataset507_coronary_arteries_cm_nativ_400subj"},
    509: {"foldername": "Dataset509_coronary_arteries_cm_nativ_400subj_SKELETON"},
    920: {"foldername": "Dataset920_aortic_sinuses_cm_nativ_400subj"},
    710: {"foldername": "Dataset710_renal_arteries_127subj"},
    713: {"foldername": "Dataset713_annulus_proper_48subj"},
    716: {"foldername": "Dataset716_aorta_true_false_lumen_171subj"},
    514: {"foldername": "Dataset514_annulus_pulmonary_70subj"},

    # XGBoost
    "body_stats": {
        "foldername": "body_stats_models_2026_03_24",
        "version": "v2.5.0-weights"
    },

    # CNN body stats: one multi-target model per modality
    "body_stats_cnn_mr": {
        "rel_path": "lightning_models",
        "foldername" : "mr_all_splitOrig_3d_resnet10_fl2_ep40_rtn1_can1",
        "version": "v2.5.0-weights",
    },
    "body_stats_cnn_ct": {
        "rel_path": "lightning_models",
        "foldername": "ct_all_splitOrig_3d_resnet10_fl2_ep40_rtn1_can1",
        "version": "v2.5.0-weights",
    },
}