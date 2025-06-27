import sys
import os
import textwrap
from typing import Union
from dataclasses import dataclass, field

from totalsegmentator.config import has_valid_license_offline, get_weights_dir
from totalsegmentator.libs import (
    old_model_cleanup,
    download_url_and_unpack,
    download_model_with_license_and_unpack,
)


WEIGHTS_URL = "https://github.com/wasserth/TotalSegmentator/releases/download"
TASKS_FAST_SUPPORT = ["total", "total_mr", "body", "body_mr"]
TASKS_FASTEST_SUPPORT = ["total", "total_mr"]
TASKS_FAST_IS_6MM = ["body", "body_mr"]
TASKS_COMMERCIAL = [
    "vertebrae_body",
    "heartchambers_highres",
    "appendicular_bones",
    "appendicular_bones_mr",
    "tissue_types",
    "tissue_types_mr",
    "tissue_4_types",
    "face",
    "face_mr",
    "brain_structures",
    "thigh_shoulder_muscles",
    "thigh_shoulder_muscles_mr",
    "coronary_arteries",
    "aortic_sinuses",
]

TASK_TO_ID = {
    # Standard tasks with no fast variants
    "total_highres_test": 957,
    "lung_vessels": 258,
    "cerebral_bleed": 150,
    "hip_implant": 260,
    "vertebrae_mr": 756,
    "pleural_pericard_effusion": 315,
    "liver_vessels": 8,
    "head_glands_cavities": 775,
    "headneck_bones_vessels": 776,
    "head_muscles": 777,
    "headneck_muscles": [778, 779],
    "oculomotor_muscles": 351,
    "lung_nodules": 913,
    "kidney_cysts": 789,
    "breasts": 527,
    "ventricle_parts": 552,
    "liver_segments": 570,
    "liver_segments_mr": 576,
    "craniofacial_structures": 115,
    "abdominal_muscles": 952,
    "test": [517],
    # Tasks with fast variants
    "total": {
        "default": [291, 292, 293, 294, 295],
        "fast": 297,
        "fastest": 298,
    },
    "total_mr": {"default": [850, 851], "fast": 852, "fastest": 853},
    "body": {"default": 299, "fast": 300},
    "body_mr": {"default": 597, "fast": 598},
    # Commercial tasks
    "vertebrae_body": 305,
    "heartchambers_highres": 301,
    "appendicular_bones": 304,
    "appendicular_bones_mr": 855,
    "tissue_types": 481,
    "tissue_types_mr": 925,
    "tissue_4_types": 485,
    "face": 303,
    "face_mr": 856,
    "brain_structures": 409,
    "thigh_shoulder_muscles": 857,
    "thigh_shoulder_muscles_mr": 857,
    "coronary_arteries": 507,
    "aortic_sinuses": 920,
}

TASK_WEIGHTS = {
    291: {
        "path": "Dataset291_TotalSegmentator_part1_organs_1559subj",
        "url": "v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
    },
    292: {
        "path": "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
        "url": "v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
    },
    293: {
        "path": "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
        "url": "v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
    },
    294: {
        "path": "Dataset294_TotalSegmentator_part4_muscles_1559subj",
        "url": "v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
    },
    295: {
        "path": "Dataset295_TotalSegmentator_part5_ribs_1559subj",
        "url": "v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip",
    },
    297: {
        "path": "Dataset297_TotalSegmentator_total_3mm_1559subj",
        "url": "v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip",
    },
    298: {
        "path": "Dataset298_TotalSegmentator_total_6mm_1559subj",
        "url": "v2.0.0-weights/Dataset298_TotalSegmentator_total_6mm_1559subj.zip",
    },
    299: {
        "path": "Dataset299_body_1559subj",
        "url": "v2.0.0-weights/Dataset299_body_1559subj.zip",
    },
    300: {
        "path": "Dataset300_body_6mm_1559subj",
        "url": "v2.0.0-weights/Dataset300_body_6mm_1559subj.zip",
    },
    775: {
        "path": "Dataset775_head_glands_cavities_492subj",
        "url": "v2.3.0-weights/Dataset775_head_glands_cavities_492subj.zip",
    },
    776: {
        "path": "Dataset776_headneck_bones_vessels_492subj",
        "url": "v2.3.0-weights/Dataset776_headneck_bones_vessels_492subj.zip",
    },
    777: {
        "path": "Dataset777_head_muscles_492subj",
        "url": "v2.3.0-weights/Dataset777_head_muscles_492subj.zip",
    },
    778: {
        "path": "Dataset778_headneck_muscles_part1_492subj",
        "url": "v2.3.0-weights/Dataset778_headneck_muscles_part1_492subj.zip",
    },
    779: {
        "path": "Dataset779_headneck_muscles_part2_492subj",
        "url": "v2.3.0-weights/Dataset779_headneck_muscles_part2_492subj.zip",
    },
    351: {
        "path": "Dataset351_oculomotor_muscles_18subj",
        "url": "v2.4.0-weights/Dataset351_oculomotor_muscles_18subj.zip",
    },
    789: {
        "path": "Dataset789_kidney_cyst_501subj",
        "url": "v2.5.0-weights/Dataset789_kidney_cyst_501subj.zip",
    },
    527: {
        "path": "Dataset527_breasts_1559subj",
        "url": "v2.5.0-weights/Dataset527_breasts_1559subj.zip",
    },
    552: {
        "path": "Dataset552_ventricle_parts_38subj",
        "url": "v2.5.0-weights/Dataset552_ventricle_parts_38subj.zip",
    },
    955: {
        "path": "Dataset955_TotalSegmentator_highres_part1_organs_110subj",
        "url": "TODO",
    },
    956: {
        "path": "Dataset956_TotalSegmentator_highres_part1_organs_cascade_110subj",
        "url": "TODO",
    },
    957: {
        "path": "Dataset957_TotalSegmentator_highres_part1_organs_cropBody_127subj",
        "url": "TODO",
    },
    850: {
        "path": "Dataset850_TotalSegMRI_part1_organs_1088subj",
        "url": "v2.5.0-weights/Dataset850_TotalSegMRI_part1_organs_1088subj.zip",
    },
    851: {
        "path": "Dataset851_TotalSegMRI_part2_muscles_1088subj",
        "url": "v2.5.0-weights/Dataset851_TotalSegMRI_part2_muscles_1088subj.zip",
    },
    852: {
        "path": "Dataset852_TotalSegMRI_total_3mm_1088subj",
        "url": "v2.5.0-weights/Dataset852_TotalSegMRI_total_3mm_1088subj.zip",
    },
    853: {
        "path": "Dataset853_TotalSegMRI_total_6mm_1088subj",
        "url": "v2.5.0-weights/Dataset853_TotalSegMRI_total_6mm_1088subj.zip",
    },
    597: {
        "path": "Dataset597_mri_body_139subj",
        "url": "v2.5.0-weights/Dataset597_mri_body_139subj.zip",
    },
    598: {
        "path": "Dataset598_mri_body_6mm_139subj",
        "url": "v2.5.0-weights/Dataset598_mri_body_6mm_139subj.zip",
    },
    756: {
        "path": "Dataset756_mri_vertebrae_1076subj",
        "url": "v2.5.0-weights/Dataset756_mri_vertebrae_1076subj.zip",
    },
    258: {
        "path": "Dataset258_lung_vessels_248subj",
        "url": "v2.0.0-weights/Dataset258_lung_vessels_248subj.zip",
    },
    200: {
        "path": "Task200_covid_challenge",
        "url": "TODO",
    },
    201: {
        "path": "Task201_covid",
        "url": "TODO",
    },
    150: {
        "path": "Dataset150_icb_v0",
        "url": "v2.0.0-weights/Dataset150_icb_v0.zip",
    },
    260: {
        "path": "Dataset260_hip_implant_71subj",
        "url": "v2.0.0-weights/Dataset260_hip_implant_71subj.zip",
    },
    315: {
        "path": "Dataset315_thoraxCT",
        "url": "v2.0.0-weights/Dataset315_thoraxCT.zip",
    },
    8: {
        "path": "Dataset008_HepaticVessel",
        "url": "v2.4.0-weights/Dataset008_HepaticVessel.zip",
    },
    913: {
        "path": "Dataset913_lung_nodules",
        "url": "v2.5.0-weights/Dataset913_lung_nodules.zip",
    },
    570: {
        "path": "Dataset570_ct_liver_segments",
        "url": "v2.5.0-weights/Dataset570_ct_liver_segments.zip",
    },
    576: {
        "path": "Dataset576_mri_liver_segments_120subj",
        "url": "v2.5.0-weights/Dataset576_mri_liver_segments_120subj.zip",
    },
    115: {
        "path": "Dataset115_mandible",
        "url": "v2.5.0-weights/Dataset115_mandible.zip",
    },
    952: {
        "path": "Dataset952_abdominal_muscles_167subj",
        "url": "v2.5.0-weights/Dataset952_abdominal_muscles_167subj.zip",
    },
    # Commercial models
    304: {"path": "Dataset304_appendicular_bones_ext_1559subj"},
    855: {"path": "Dataset855_TotalSegMRI_appendicular_bones_1088subj"},
    301: {"path": "Dataset301_heart_highres_1559subj"},
    303: {"path": "Dataset303_face_1559subj"},
    481: {"path": "Dataset481_tissue_1559subj"},
    485: {"path": "Dataset485_tissue_4types_1559subj"},
    305: {"path": "Dataset305_vertebrae_discs_1559subj"},
    925: {"path": "Dataset925_MRI_tissue_subset_903subj"},
    856: {"path": "Dataset856_TotalSegMRI_face_1088subj"},
    409: {"path": "Dataset409_neuro_550subj"},
    857: {"path": "Dataset857_TotalSegMRI_thigh_shoulder_1088subj"},
    507: {"path": "Dataset507_coronary_arteries_cm_nativ_400subj"},
    920: {"path": "Dataset920_aortic_sinuses_cm_nativ_400subj"},
}

@dataclass
class Task:
    task: str = field(default=None)
    task_id: int = field(default=None)
    resample: Union[float, list[float]] = field(default=1.5)
    trainer: str = field(default="nnUNetTrainer")
    crop: Union[list[str], None] = field(default=None)
    model: str = field(default="3d_fullres")
    cascade: bool = field(default=None)
    folds: list[int] = field(default_factory=lambda: [0])
    crop_addon: list[int] = field(default_factory=lambda: [3, 3, 3])

    def __post_init__(self):
        self.supports_fast = self.task in TASKS_FAST_SUPPORT
        self.supports_fastest = self.task in TASKS_FASTEST_SUPPORT

        if self.task in TASKS_COMMERCIAL:
            show_license_info()

    def download_weights(self):
        config_dir = get_weights_dir()
        config_dir.mkdir(exist_ok=True, parents=True)

        if self.task_id not in TASK_WEIGHTS:
            raise ValueError(f"Task {self.task_id} not found.")
        self.weights_path = config_dir / TASK_WEIGHTS[self.task_id]["path"]

        if not os.path.exists(self.weights_path):
            old_model_cleanup(config_dir)
            if self.task in TASKS_COMMERCIAL:
                print(
                    f"Downloading weights for commercial task {self.task_id}..."
                )
                download_model_with_license_and_unpack(self.task, config_dir)
                return
            if "url" not in TASK_WEIGHTS[self.task_id]:
                raise ValueError(
                    f"Task {self.task_id} is commercial and automatic download is not supported."
                )
            elif TASK_WEIGHTS[self.task_id]["url"] == "TODO":
                raise ValueError(
                    f"Task {self.task_id} weights not available yet."
                )
            self.weights_url = (
                f"{WEIGHTS_URL}/{TASK_WEIGHTS[self.task_id]['url']}"
            )
            print(f"Downloading weights for task {self.task_id}...")
            download_url_and_unpack(self.weights_url, config_dir)

    def check_fast(self, fast: bool, fastest: bool, quiet: bool):
        if not self.supports_fast and fast:
            raise ValueError(
                f"Task {self.task_id} does not support the --fast option."
            )
        if not self.supports_fastest and fastest:
            raise ValueError(
                f"Task {self.task_id} does not support the --fastest option."
            )
        if fast:
            self.task_id = self.task_id
            if not quiet:
                if self.task in TASKS_FAST_IS_6MM:
                    print(
                        "Using 'fast' option: resampling to lower resolution (6mm)"
                    )
                else:
                    print(
                        "Using 'fast' option: resampling to lower resolution (3mm)"
                    )
        if fastest:
            self.task_id = self.task_id
            if not quiet:
                print(
                    "Using 'fastest' option: resampling to lower resolution (6mm)"
                )


def show_license_info() -> None:
    """
    Show license info for commercial models.
    """
    status, message = has_valid_license_offline()
    if status == "missing_license":
        # textwarp needed to remove the indentation of the multiline string
        print(
            textwrap.dedent(
                """\
              In contrast to the other tasks this task is not openly available.
              It requires a license. For non-commercial usage a free license can be
              acquired here:
              https://backend.totalsegmentator.com/license-academic/

              For commercial usage contact: jakob.wasserthal@usb.ch
              """
            )
        )
        sys.exit(1)
    elif status == "invalid_license":
        print(message)
        sys.exit(1)
    elif status == "missing_config_file":
        print(message)
        sys.exit(1)


def get_task(task: str, fast: bool, fastest: bool, quiet: bool = True) -> Task:
    task_id = TASK_TO_ID.get(task)
    if task_id is None:
        raise ValueError(f"Unknown task: {task}")
    if isinstance(task_id, dict):
        if fast:
            task_id = task_id["fast"]
        elif fastest:
            task_id = task_id["fastest"]
        else:
            task_id = task_id["default"]
    if task == "total":
        if fast:
            output_task = Task(
                task=task,
                task_id=task_id,
                resample=3.0,
                trainer="nnUNetTrainer_4000epochs_NoMirroring",
            )
        elif fastest:
            output_task = Task(
                task=task,
                task_id=task_id,
                resample=6.0,
                trainer="nnUNetTrainer_4000epochs_NoMirroring",
            )
        else:
            output_task = Task(
                task=task,
                task_id=task_id,
                trainer="nnUNetTrainerNoMirroring",
            )

    elif task == "total_highres_test":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.75, 0.75, 1.0],
            trainer="nnUNetTrainerNoMirroring",
            model="3d_fullres_high",
            cascade=False,
        )
    elif task == "total_mr":
        if fast:
            output_task = Task(
                task=task,
                task_id=task_id,
                resample=3.0,
                trainer="nnUNetTrainer_2000epochs_NoMirroring",
            )
        elif fastest:
            output_task = Task(
                task=task,
                task_id=task_id,
                resample=6.0,
                trainer="nnUNetTrainer_2000epochs_NoMirroring",
            )
        else:
            output_task = Task(
                task=task,
                task_id=task_id,
                trainer="nnUNetTrainer_2000epochs_NoMirroring",
            )

    elif task == "lung_vessels":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=None,
            crop=[
                "lung_upper_lobe_left",
                "lung_lower_lobe_left",
                "lung_upper_lobe_right",
                "lung_middle_lobe_right",
                "lung_lower_lobe_right",
            ],
        )
    elif task == "cerebral_bleed":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=None,
            crop=["brain"],
        )
    elif task == "hip_implant":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=None,
            crop=["femur_left", "femur_right", "hip_left", "hip_right"],
        )

    elif task == "body":
        if fast:
            output_task = Task(
                task=task,
                task_id=task_id,
                resample=6.0,
            )
        else:
            output_task = Task(
                task=task,
                task_id=task_id,
            )
    elif task == "body_mr":
        if fast:
            output_task = Task(
                task=task,
                task_id=task_id,
                resample=6.0,
                trainer="nnUNetTrainer_DASegOrd0",
            )
        else:
            output_task = Task(
                task=task,
                task_id=task_id,
                trainer="nnUNetTrainer_DASegOrd0",
            )
    elif task == "vertebrae_mr":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
        )
    elif task == "pleural_pericard_effusion":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=None,
            crop=[
                "lung_upper_lobe_left",
                "lung_lower_lobe_left",
                "lung_upper_lobe_right",
                "lung_middle_lobe_right",
                "lung_lower_lobe_right",
            ],
            crop_addon=[50, 50, 50],
            folds=None,
        )
    elif task == "liver_vessels":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=None,
            crop=["liver"],
            crop_addon=[20, 20, 20],
        )
    elif task == "head_glands_cavities":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.75, 0.75, 1.0],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["skull"],
            crop_addon=[10, 10, 10],
            model="3d_fullres_high",
        )
    elif task == "headneck_bones_vessels":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.75, 0.75, 1.0],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=[
                "clavicula_left",
                "clavicula_right",
                "vertebrae_C1",
                "vertebrae_C5",
                "vertebrae_T1",
                "vertebrae_T4",
            ],
            crop_addon=[40, 40, 40],
            model="3d_fullres_high",
        )
    elif task == "head_muscles":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.75, 0.75, 1.0],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["skull"],
            crop_addon=[10, 10, 10],
            model="3d_fullres_high",
        )
    elif task == "headneck_muscles":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.75, 0.75, 1.0],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=[
                "clavicula_left",
                "clavicula_right",
                "vertebrae_C1",
                "vertebrae_C5",
                "vertebrae_T1",
                "vertebrae_T4",
            ],
            crop_addon=[40, 40, 40],
            model="3d_fullres_high",
        )

    elif task == "oculomotor_muscles":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[
                0.47251562774181366,
                0.47251562774181366,
                0.8500002026557922,
            ],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["skull"],
            crop_addon=[20, 20, 20],
        )
    elif task == "lung_nodules":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring",
            crop=[
                "lung_upper_lobe_left",
                "lung_lower_lobe_left",
                "lung_upper_lobe_right",
                "lung_middle_lobe_right",
                "lung_lower_lobe_right",
            ],
            crop_addon=[10, 10, 10],
        )
    elif task == "kidney_cysts":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["kidney_left", "kidney_right", "liver", "spleen", "colon"],
            crop_addon=[10, 10, 10],
        )

    elif task == "breasts":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
        )
    elif task == "ventricle_parts":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[1.0, 0.4345703125, 0.4384765625],
            trainer="nnUNetTrainerNoMirroring",
            crop=["brain"],
            crop_addon=[0, 0, 0],
        )
    elif task == "liver_segments":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[1.5, 0.8046879768371582, 0.8046879768371582],
            trainer="nnUNetTrainerNoMirroring",
            crop=["liver"],
            crop_addon=[10, 10, 10],
        )
    elif task == "liver_segments_mr":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[3.0, 1.1875, 1.1250001788139343],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["liver"],
            crop_addon=[10, 10, 10],
        )
    elif task == "craniofacial_structures":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.5, 0.5, 0.5],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["skull"],
            crop_addon=[20, 20, 20],
        )
    # This model only segments within T4-L4. In training only labels in this region were annotated. Therefore,
    # I do not have to crop to this region, but model automatically only predicts in this region.
    elif task == "abdominal_muscles":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.75, 0.75, 1],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["body_trunc"],
            crop_addon=[5, 5, 5],
            model="3d_fullres_high",
        )

    # Commercial models
    elif task == "vertebrae_body":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_DASegOrd0",
        )
    elif task == "heartchambers_highres":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=None,
            crop=["heart"],
            crop_addon=[5, 5, 5],
        )
    elif task == "appendicular_bones":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainerNoMirroring",
        )
    elif task == "appendicular_bones_mr":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_2000epochs_NoMirroring",
        )
    elif task == "tissue_types":
        output_task = Task(task=task, task_id=481)
    elif task == "tissue_types_mr":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
        )
    elif task == "tissue_4_types":
        output_task = Task(task=task, task_id=485)
    elif task == "face":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainerNoMirroring",
        )
    elif task == "face_mr":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_2000epochs_NoMirroring",
        )
    elif task == "brain_structures":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.5, 0.5, 1.0],
            trainer="nnUNetTrainer_DASegOrd0",
            crop=["brain"],
            crop_addon=[10, 10, 10],
            model="3d_fullres_high",
        )

    elif task == "thigh_shoulder_muscles":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_2000epochs_NoMirroring",
        )
    elif task == "thigh_shoulder_muscles_mr":
        output_task = Task(
            task=task,
            task_id=task_id,
            trainer="nnUNetTrainer_2000epochs_NoMirroring",
        )

    elif task == "coronary_arteries":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.7, 0.7, 0.7],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["heart"],
            crop_addon=[20, 20, 20],
            model="3d_fullres_high",
        )

    elif task == "aortic_sinuses":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=[0.7, 0.7, 0.7],
            trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
            crop=["heart"],
            crop_addon=[0, 0, 0],
            model="3d_fullres_high",
        )

    elif task == "test":
        output_task = Task(
            task=task,
            task_id=task_id,
            resample=None,
            trainer="nnUNetTrainerV2",
            crop="body",
        )

    output_task.check_fast(fast, fastest, quiet)
    output_task.download_weights()
    return output_task
