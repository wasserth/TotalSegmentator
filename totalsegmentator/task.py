import sys
import textwrap
from typing import Union
from totalsegmentator.config import has_valid_license_offline
from dataclasses import dataclass, field

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


def get_task(task: str, fast: bool, fastest: bool, quiet: bool) -> Task:
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
    return output_task
