import os
import sys
import time
import shutil
import zipfile
from pathlib import Path
import subprocess
import platform
import importlib.metadata
import importlib

from tqdm import tqdm
import numpy as np
import nibabel as nib
import dicom2nifti

from totalsegmentator.config import get_weights_dir
from totalsegmentator.dicom_utils import rgb_to_cielab_dicom, generate_random_color, load_snomed_mapping, load_color_mapping


def command_exists(command):
    return shutil.which(command) is not None


def download_dcm2niix():
    import urllib.request
    print("  Downloading dcm2niix...")

    if platform.system() == "Windows":
        # url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_win.zip"
        url = "https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_win.zip"
    elif platform.system() == "Darwin":  # Mac
        # raise ValueError("For MacOS automatic installation of dcm2niix not possible. Install it manually.")
        if platform.machine().startswith("arm") or platform.machine().startswith("aarch"):  # arm
            # url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/macos_dcm2niix.pkg"
            url = "https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_macos.zip"
        else:  # intel
            # unclear if this is the right link (is the same as for arm)
            # url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/macos_dcm2niix.pkg"
            url = "https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_macos.zip"
    elif platform.system() == "Linux":
        # url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip"
        url = "https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_lnx.zip"
    else:
        raise ValueError("Unknown operating system. Can not download the right version of dcm2niix.")

    config_dir = get_weights_dir()

    urllib.request.urlretrieve(url, config_dir / "dcm2niix.zip")
    with zipfile.ZipFile(config_dir / "dcm2niix.zip", 'r') as zip_ref:
        zip_ref.extractall(config_dir)

    # Give execution permission to the script
    if platform.system() == "Windows":
        os.chmod(config_dir / "dcm2niix.exe", 0o755)
    else:
        os.chmod(config_dir / "dcm2niix", 0o755)

    # Clean up
    if (config_dir / "dcm2niix.zip").exists():
        os.remove(config_dir / "dcm2niix.zip")
    if (config_dir / "dcm2niibatch").exists():
        os.remove(config_dir / "dcm2niibatch")


def dcm_to_nifti_LEGACY(input_path, output_path, verbose=False):
    """
    Uses dcm2niix (does not properly work on windows)

    input_path: a directory of dicom slices
    output_path: a nifti file path
    """
    verbose_str = "" if verbose else "> /dev/null"

    config_dir = get_weights_dir()

    if command_exists("dcm2niix"):
        dcm2niix = "dcm2niix"
    else:
        if platform.system() == "Windows":
            dcm2niix = config_dir / "dcm2niix.exe"
        else:
            dcm2niix = config_dir / "dcm2niix"
        if not dcm2niix.exists():
            download_dcm2niix()

    subprocess.call(f"\"{dcm2niix}\" -o {output_path.parent} -z y -f {output_path.name[:-7]} {input_path} {verbose_str}", shell=True)

    if not output_path.exists():
        print(f"Content of dcm2niix output folder ({output_path.parent}):")
        print(list(output_path.parent.glob("*")))
        raise ValueError("dcm2niix failed to convert dicom to nifti.")

    nii_files = list(output_path.parent.glob("*.nii.gz"))

    if len(nii_files) > 1:
        print("WARNING: Dicom to nifti resulted in several nifti files. Skipping files which contain ROI in filename.")
        for nii_file in nii_files:
            # output file name is "converted_dcm.nii.gz" so if ROI in name, then this can be deleted
            if "ROI" in nii_file.name:
                os.remove(nii_file)
                print(f"Skipped: {nii_file.name}")

    nii_files = list(output_path.parent.glob("*.nii.gz"))

    if len(nii_files) > 1:
        print("WARNING: Dicom to nifti resulted in several nifti files. Only using first one.")
        print([f.name for f in nii_files])
        for nii_file in nii_files[1:]:
            os.remove(nii_file)
        # todo: have to rename first file to not contain any counter which is automatically added by dcm2niix

    os.remove(str(output_path)[:-7] + ".json")


def _infer_plane_from_iop(iop):
    """Infer image plane (axial/coronal/sagittal/oblique) from ImageOrientationPatient.
    iop: list/tuple of 6 floats [rx, ry, rz, cx, cy, cz] for row and column direction cosines.
    Returns one of 'axial', 'coronal', 'sagittal', or 'oblique'.
    """
    try:
        import math
        rx, ry, rz, cx, cy, cz = (float(x) for x in iop)
        # Normal = row x col
        nx = ry * cz - rz * cy
        ny = rz * cx - rx * cz
        nz = rx * cy - ry * cx
        # Compare absolute components of normal to axes
        ax = abs(nx)
        ay = abs(ny)
        az = abs(nz)
        # A simple threshold to decide planarity vs oblique
        # If the normal is mostly along one axis, we consider it that plane
        dominant = max(ax, ay, az)
        if dominant < 0.9:
            return 'oblique'
        if az == dominant:
            return 'axial'
        if ay == dominant:
            return 'coronal'
        return 'sagittal'
    except Exception:
        return 'oblique'


def _extract_orientation_from_datasets(datasets):
    """Extract orientation metadata from a list of already loaded pydicom datasets.
    Returns a dict with keys similar to extract_orientation_metadata.
    """
    try:
        if not datasets:
            return None
        first = datasets[0]
        iop = list(getattr(first, 'ImageOrientationPatient', []))
        # Collect first few IPP values
        ipp_list = []
        for ds in datasets[:5]:
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp is not None:
                ipp_list.append([float(x) for x in ipp])
        slice_spacing = None
        if len(iop) == 6 and len(ipp_list) >= 2:
            try:
                rx, ry, rz, cx, cy, cz = (float(x) for x in iop)
                nx = ry * cz - rz * cy
                ny = rz * cx - rx * cz
                nz = rx * cy - ry * cx
                p0 = np.array(ipp_list[0], dtype=float)
                p1 = np.array(ipp_list[1], dtype=float)
                diff = p1 - p0
                slice_spacing = float(abs(diff[0]*nx + diff[1]*ny + diff[2]*nz))
            except Exception:
                slice_spacing = None
        pixel_spacing = None
        try:
            ps = getattr(first, 'PixelSpacing', None)
            if ps is not None:
                pixel_spacing = [float(ps[0]), float(ps[1])]
        except Exception:
            pixel_spacing = None
        modality = getattr(first, 'Modality', None)
        plane = _infer_plane_from_iop(iop) if len(iop) == 6 else None
        return {
            "modality": str(modality) if modality else None,
            "image_orientation_patient": iop if len(iop) == 6 else None,
            "image_position_patient_first": ipp_list[0] if ipp_list else None,
            "pixel_spacing": pixel_spacing,
            "slice_thickness": float(getattr(first, 'SliceThickness', 0)) or None,
            "slice_spacing": slice_spacing,
            "plane": plane
        }
    except Exception:
        return None


def dcm_to_nifti(input_path, output_path, tmp_dir=None, verbose=False):
    """
    Uses dicom2nifti package (also works on windows)

    input_path: a directory of dicom slices or a zip file of dicom slices or a bytes object of zip file
    output_path: a nifti file path
    tmp_dir: extract zip file to this directory, else to the same directory as the zip file. Needs to be set if input is a zip file.
    """
    # Check if input_path is a zip file and extract it
    if zipfile.is_zipfile(input_path):
        if tmp_dir is None:
            raise ValueError("tmp_dir must be set when input_path is a zip file or bytes object of zip file")
        if verbose: print(f"Extracting zip file: {input_path}")
        extract_dir = os.path.splitext(input_path)[0] if tmp_dir is None else tmp_dir / "extracted_dcm"
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            input_path = extract_dir

    # Convert to nifti
    dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=True)


def detect_dicom_modality(series_path: Path) -> str | None:
    """Return DICOM Modality from a series directory (e.g., 'CT' or 'MR').
    Looks for any file in the directory and reads its Modality tag using pydicom.
    Returns None if not found or unreadable.
    """
    try:
        import pydicom
    except Exception:
        return None
    try:
        p = Path(series_path)
        if p.is_file():
            candidates = [p]
        else:
            # Prefer .dcm files but fall back to any file
            dcm_files = sorted(p.glob("*.dcm"))
            candidates = dcm_files if dcm_files else [f for f in p.iterdir() if f.is_file()]
        for f in candidates:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                modality = getattr(ds, 'Modality', None)
                if modality:
                    return str(modality)
            except Exception:
                continue
    except Exception:
        return None
    return None


def save_mask_as_rtstruct(img_data, selected_classes, dcm_reference_file, output_path):
    """
    dcm_reference_file: a directory with dcm slices ??
    """
    from rt_utils import RTStructBuilder
    import logging
    logging.basicConfig(level=logging.WARNING)  # avoid messages from rt_utils

    # Load deterministic color mapping (same as used for DICOM SEG)
    try:
        color_map = load_color_mapping()
    except Exception:
        color_map = {}

    # create new RT Struct - requires original DICOM
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_reference_file)
    orientation_metadata = _extract_orientation_from_datasets(rtstruct.series_data)

    # add mask to RT Struct
    for class_idx, class_name in tqdm(selected_classes.items()):
        binary_img = img_data == class_idx
        if binary_img.sum() > 0:  # only save none-empty images

            # rotate nii to match DICOM orientation
            if orientation_metadata.get("plane") == "axial":
                binary_img = np.transpose(binary_img, (1, 0, 2))[::-1, :, :]
            elif orientation_metadata.get("plane") == "coronal":
                binary_img = np.transpose(binary_img, (2, 0, 1))[::-1, :, ::-1]
            elif orientation_metadata.get("plane") == "sagittal":
                binary_img = np.transpose(binary_img, (2, 1, 0))[::-1, ::-1, ::-1]
            else:
                raise ValueError(
                    f"Segmentation shape {binary_img.shape} does not match DICOM dimensions. Cannot create DICOM SEG with mismatched dimensions."
                )

            # Determine RGB color: deterministic if available, else fallback vibrant random (same logic as DICOM SEG)
            rgb_color = color_map.get(class_name)
            if rgb_color is None:
                rgb_color = generate_random_color()

            # add segmentation to RT Struct with color
            rtstruct.add_roi(
                mask=binary_img,  # has to be a binary numpy array
                name=class_name,
                color=list(rgb_color)
            )

    rtstruct.save(str(output_path))


def save_mask_as_dicomseg(img_data, selected_classes, dcm_reference_file, output_path, nifti_affine, use_gpu=False):
    """
    Save segmentation as DICOM SEG using highdicom library.

    This version optionally accelerates heavy array operations (mask creation, stacking,
    flipping, transposing) using CuPy when available. Final pixel data is converted back
    to NumPy before passing to highdicom.
    
    Args:
        img_data: segmentation data (multilabel image)
        selected_classes: dict mapping class indices to class names
        dcm_reference_file: a directory with dcm slices
        output_path: output path for the DICOM SEG file  (file or directory)
        nifti_affine: affine transformation matrix from nifti image
        use_gpu: whether to use GPU for heavy array operations
    """
    import highdicom as hd
    import pydicom
    from pydicom.sr.codedict import codes
    from highdicom.seg.content import SegmentDescription

    # Try to import CuPy; fall back to NumPy if unavailable
    try:
        cp = importlib.import_module("cupy")  # dynamic import, may fail
        cupy_available = True and use_gpu
    except Exception:
        cp = None
        cupy_available = False

    # Decide computation backend (xp is either numpy or cupy)
    if cupy_available:
        xp = cp
        # Ensure data is on GPU
        if not isinstance(img_data, cp.ndarray):
            img_data = cp.asarray(img_data)
    else:
        xp = np
    
    # Get TotalSegmentator version
    version = importlib.metadata.version("TotalSegmentator")
    
    # Load SNOMED CT codes mapping
    snomed_map = load_snomed_mapping()
    color_map = load_color_mapping()  # deterministic colors
    
    # Read reference DICOM series
    dcm_files = sorted(Path(dcm_reference_file).glob("*.dcm"))
    if len(dcm_files) == 0:
        # Try without extension
        dcm_files = sorted([f for f in Path(dcm_reference_file).iterdir() if f.is_file()])
    
    if len(dcm_files) == 0:
        raise ValueError(f"No DICOM files found in {dcm_reference_file}")
    
    # Load all DICOM slices
    source_images = [pydicom.dcmread(str(f)) for f in dcm_files]

    # Derive orientation metadata directly from loaded datasets
    orientation_metadata = _extract_orientation_from_datasets(source_images)

    # Validate and fix SOPClassUID if missing or empty
    for img in source_images:
        if not hasattr(img, 'SOPClassUID') or img.SOPClassUID == '':
            # Set to CT Image Storage by default (most common for TotalSegmentator)
            # If it's an MR image, we could check Modality tag to determine the correct UID
            if hasattr(img, 'Modality'):
                if img.Modality == 'CT':
                    img.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
                elif img.Modality == 'MR':
                    img.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
                elif img.Modality == 'PT':
                    img.SOPClassUID = '1.2.840.10008.5.1.4.1.1.128'  # PET Image Storage
                else:
                    # Default to Secondary Capture Image Storage for unknown modalities
                    img.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
            else:
                # If no Modality tag, default to CT Image Storage
                img.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage

    # Dimensions of the DICOM image grid
    dcm_rows = int(source_images[0].Rows)
    dcm_cols = int(source_images[0].Columns)
    dcm_slices = len(source_images)

    # The segmentation comes in NIfTI orientation, need to reorient to match DICOM
    # NIfTI typically has shape that might need transposing to match DICOM rows x cols x slices
    seg_shape = img_data.shape

    # Orientation handling based on DICOM dimensions
    if seg_shape == (dcm_cols, dcm_rows, dcm_slices) and orientation_metadata.get("plane") == "axial":
        img_data = xp.transpose(img_data, (1, 0, 2))
        # Sort by Image Position Patient
        if hasattr(source_images[0], 'ImagePositionPatient'):
            source_images = sorted(source_images, key=lambda x: float(x.ImagePositionPatient[2]))

    elif seg_shape == (dcm_cols, dcm_slices, dcm_rows) and orientation_metadata.get("plane") == "coronal":
        img_data = xp.transpose(img_data, (0, 2, 1))

        # Sort by Image Position Patient
        if hasattr(source_images[0], 'ImagePositionPatient'):
            source_images = sorted(source_images, key=lambda x: float(x.ImagePositionPatient[1]))

    elif seg_shape == (dcm_slices, dcm_rows, dcm_cols) and orientation_metadata.get("plane") == "sagittal":
        img_data = xp.transpose(img_data, (1, 2, 0))

        # Sort by Image Position Patient
        if hasattr(source_images[0], 'ImagePositionPatient'):
            source_images = sorted(source_images, key=lambda x: float(x.ImagePositionPatient[0]))

    elif seg_shape != (dcm_rows, dcm_cols, dcm_slices):
        raise ValueError(
            f"Segmentation shape {seg_shape} does not match DICOM dimensions ({dcm_rows}, {dcm_cols}, {dcm_slices}). Cannot create DICOM SEG with mismatched dimensions."
        )

    # Check if remapping of labels is needed
    inferred_classes = xp.unique(img_data)
    need_remap = img_data.max() != (len(inferred_classes) - 1)  # labels not contiguous (excluding background 0)
    remapped_image = xp.zeros_like(img_data) if need_remap else None

    segment_descriptions = []     # Prepare segment descriptions

    for class_idx, class_name in tqdm(selected_classes.items(), desc="Preparing segments"):
        binary_mask = img_data == class_idx

        if binary_mask.sum() == 0:  # skip empty segments
            continue

        if class_name in snomed_map:
            snomed = snomed_map[class_name]
            property_category = hd.sr.CodedConcept(
                value=snomed['property_category']['value'],
                scheme_designator=snomed['property_category']['scheme'],
                meaning=snomed['property_category']['meaning']
            )
            property_type = hd.sr.CodedConcept(
                value=snomed['property_type']['value'],
                scheme_designator=snomed['property_type']['scheme'],
                meaning=snomed['property_type']['meaning']
            )
            segment_desc = SegmentDescription(
                segment_number=len(segment_descriptions) + 1,
                segment_label=class_name,
                segmented_property_category=property_category,
                segmented_property_type=property_type,
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=hd.AlgorithmIdentificationSequence(
                    name="TotalSegmentator",
                    version=version,
                    family=codes.DCM.ArtificialIntelligence
                )
            )
            segment_desc.RecommendedDisplayCIELabValue = list(rgb_to_cielab_dicom(color_map.get(class_name)))
        else:
            segment_desc = SegmentDescription(
                segment_number=len(segment_descriptions) + 1,
                segment_label=class_name,
                segmented_property_category=codes.SCT.Tissue,
                segmented_property_type=codes.SCT.Tissue,
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=hd.AlgorithmIdentificationSequence(
                    name="TotalSegmentator",
                    version=version,
                    family=codes.DCM.ArtificialIntelligence
                )
            )
            segment_desc.RecommendedDisplayCIELabValue = list(rgb_to_cielab_dicom(generate_random_color()))

        segment_descriptions.append(segment_desc)

        if need_remap:
            remapped_image[binary_mask] = segment_desc.SegmentNumber

    if len(segment_descriptions) == 0:
        raise ValueError("No non-empty segments found to save")

    if need_remap:
        img_data = remapped_image

    # 1) DICOM has orientation (rows, cols, slices), while HIGHDICOM expects (slices, rows, cols)
    # 2) Flip along x and z axes to correct coordinate system difference between DICOM and HIGHDICOM
    if orientation_metadata.get("plane") == "axial":
        img_data = xp.transpose(img_data, (2, 0, 1))[:, ::-1, :]
    elif orientation_metadata.get("plane") == "coronal":
        img_data = xp.transpose(img_data, (2, 1, 0))[::-1, ::-1, :]
    elif orientation_metadata.get("plane") == "sagittal":
        img_data = xp.transpose(img_data, (2, 1, 0))[:, ::-1, ::-1]

    # Convert back to NumPy array if using CuPy
    if cupy_available and isinstance(img_data, cp.ndarray):
        img_data = cp.asnumpy(img_data)

    series_instance_uid = hd.UID()
    # Create DICOM SEG
    # Note: highdicom will handle the proper encoding of the multi-frame structure
    seg = hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=img_data,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=series_instance_uid,
        series_number=100,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="TotalSegmentator",
        manufacturer_model_name="TotalSegmentator",
        software_versions=version,
        device_serial_number="1"
    )

    if output_path.is_dir():
        output_path = output_path / f"{series_instance_uid}.dcm"
    
    # Save DICOM SEG file with explicit endianness and VR settings to avoid deprecation warnings
    seg.save_as(str(output_path), little_endian=True, implicit_vr=False)


