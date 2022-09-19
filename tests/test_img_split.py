from pathlib import Path

import numpy as np
import nibabel as nib


tmp_dir = Path("/home/jakob/Downloads/nnunet_test/parts_test")

img_in_rsp = nib.load(tmp_dir / "ct3mm_0000.nii.gz")

third = img_in_rsp.shape[2] // 3
margin = 20
img_in_rsp_data = img_in_rsp.get_fdata()
nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third+margin], img_in_rsp.affine),
        tmp_dir / "s01_0000.nii.gz")
nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third+1-margin:third*2+margin], img_in_rsp.affine),
        tmp_dir / "s02_0000.nii.gz")
nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third*2+1-margin:], img_in_rsp.affine),
        tmp_dir / "s03_0000.nii.gz")


combined_img = np.zeros(img_in_rsp.shape, dtype=np.int32)
# print(f"s01.shape: {nib.load(tmp_dir / 's01.nii.gz').shape}")
# print(third)
combined_img[:,:,:third] = nib.load(tmp_dir / "s01_0000.nii.gz").get_fdata()[:,:,:-margin]
# print(f"s02.shape: {nib.load(tmp_dir / 's02.nii.gz').shape}")
# print(margin)
# print(-margin)
combined_img[:,:,third:third*2] = nib.load(tmp_dir / "s02_0000.nii.gz").get_fdata()[:,:,margin-1:-margin]
# print(f"s03.shape: {nib.load(tmp_dir / 's03.nii.gz').shape}")
# print(margin)
combined_img[:,:,third*2:] = nib.load(tmp_dir / "s03_0000.nii.gz").get_fdata()[:,:,margin-1:]
nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "ct3mm_0000_combined.nii.gz")