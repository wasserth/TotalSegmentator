import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring

class nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring(nnUNetTrainerNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 2.5e-3
