import sys
from pathlib import Path

import json
import time
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import binary_erosion as scipy_binary_erosion
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.spine_report.utils import get_erosion_struct_elem

# Constants
REGION_ORDER = {'C': 1, 'T': 2, 'L': 3, 'S': 4}
MEASUREMENT_POSITIONS = [0.15, 0.5, 0.85]  # anterior keep a bit more margin, because there often more spiking?
# INFO: using smoothed mask for horizontal axis endpoints does not work so well. Do not use for now.
POSITION_NAMES = [f"{int(round(pos * 100))}%" for pos in MEASUREMENT_POSITIONS]
POSITION_MAPPING = {POSITION_NAMES[0]: 'posterior', POSITION_NAMES[1]: 'middle', POSITION_NAMES[2]: 'anterior'}
FRACTURE_THRESHOLD = 0.8
HU_WINDOW = (-500, 1500)
CROP_SIZE_MM = 80
PADDING_MM = 40
TEXT_OFFSET_MM = -20
STEP_SIZE = 0.5
BODY_VOLUME_YELLOW_THRESHOLD_ML = 2.0
# do not make too big, because already smaller because only doing erosion and not dilation
# on the other hand: more robust if bigger; but also might not find all fractures
SCAN_MARGIN_RATIO = 0.25  # fraction of x-extent to discard at both ends when scanning slices



class VertebraeProcessor:
    """Handles vertebrae analysis and measurement"""
    
    def __init__(self, ct_img, verte_img, verte_label_map, verte_body_img, verte_body_label_map, horizontal_mask_type='smoothed', spine_range='all'):
        self.ct_img = ct_img
        self.verte_img = verte_img
        self.verte_body_img = verte_body_img
        self.spacing = ct_img.header.get_zooms()
        # Control whether horizontal axis endpoints use smoothed or raw mask
        self.horizontal_mask_type = horizontal_mask_type
        self.spine_range = spine_range
        
        self.ct_data = ct_img.get_fdata()
        self.verte_data = verte_img.get_fdata()
        self.verte_body_data = verte_body_img.get_fdata()
        
        self.vertebrae_labels = self._extract_vertebrae_labels(verte_label_map)
        self.body_mask = self._get_body_mask(verte_body_label_map)
        self.sorted_vertebrae = self._sort_vertebrae()
        # Storage for optional debug export of smoothed per-vertebra masks
        self.smoothed_body_multilabel = np.zeros(self.verte_data.shape, dtype=np.uint16)
        
    def _extract_vertebrae_labels(self, label_map):
        return {k: v for k, v in label_map.items() if v.startswith('vertebrae_')}
    
    def _vertebra_sort_key(self, label_name):
        parts = label_name.replace('vertebrae_', '').strip()
        if len(parts) >= 2:
            region = parts[0]
            try:
                number = int(parts[1:])
                return (REGION_ORDER.get(region, 5), number)
            except ValueError:
                pass
        return (99, 0)
    
    def _sort_vertebrae(self):
        return sorted(self.vertebrae_labels.items(), key=lambda x: self._vertebra_sort_key(x[1]))
    
    def _filter_vertebrae_by_range(self):
        """Filter vertebrae based on spine_range, keeping adjacent ones for centroid calculation.
        
        Returns:
            filtered_vertebrae: list of (label_id, label_name) to process
            helper_vertebrae: set of label_names that are only for centroid calculation
        """
        if self.spine_range == 'all':
            return self.sorted_vertebrae, set()
        
        # Define the range mapping
        range_mapping = {
            'thoracic_lumbar': lambda name: name.startswith('vertebrae_T') or name.startswith('vertebrae_L'),
            'lumbar': lambda name: name.startswith('vertebrae_L'),
            'l1_l4': lambda name: name in ['vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4']
        }
        
        range_filter = range_mapping.get(self.spine_range)
        if not range_filter:
            print(f"Warning: Unknown spine_range '{self.spine_range}', using 'all'")
            return self.sorted_vertebrae, set()
        
        # Find the vertebrae in the selected range
        in_range_indices = [i for i, (_, name) in enumerate(self.sorted_vertebrae) if range_filter(name)]
        
        if not in_range_indices:
            print(f"Warning: No vertebrae found for spine_range '{self.spine_range}'")
            return [], set()
        
        first_idx = in_range_indices[0]
        last_idx = in_range_indices[-1]
        
        # Include one vertebra before and after (if available)
        start_idx = max(0, first_idx - 1)
        end_idx = min(len(self.sorted_vertebrae) - 1, last_idx + 1)
        
        filtered_vertebrae = self.sorted_vertebrae[start_idx:end_idx + 1]
        
        # Mark vertebrae outside the original range as helpers (only for centroid calculation)
        helper_vertebrae = set()
        for i in range(start_idx, end_idx + 1):
            _, name = self.sorted_vertebrae[i]
            if i < first_idx or i > last_idx:
                helper_vertebrae.add(name)
                print(f"Using {name} as helper for centroid calculation only")
        
        return filtered_vertebrae, helper_vertebrae
    
    def _get_body_mask(self, body_label_map):
        if not body_label_map:
            print("Warning: No vertebrae body mask found, using full vertebrae masks")
            return np.ones_like(self.verte_data, dtype=bool)
            
        for body_id, body_name in body_label_map.items():
            if 'vertebrae_body' in body_name or 'body' in body_name.lower():
                print(f"Using body mask: {body_name} (ID: {body_id})")
                return self.verte_body_data == body_id
        
        print("Warning: No vertebrae body mask found, using full vertebrae masks")
        return np.ones_like(self.verte_data, dtype=bool)
    
    def _calculate_centroid(self, mask):
        coords = np.where(mask)
        return np.array([np.mean(coords[i]) for i in range(3)]) if len(coords[0]) > 0 else None
    
    def _calculate_axes(self, centroids, index):
        prev_centroid = centroids[index-1] if index > 0 else None
        next_centroid = centroids[index+1] if index < len(centroids)-1 else None
        current_centroid = centroids[index]
        
        # Calculate height axis
        if prev_centroid is not None and next_centroid is not None:
            height_axis = next_centroid - prev_centroid
        elif prev_centroid is not None:
            height_axis = current_centroid - prev_centroid
        elif next_centroid is not None:
            height_axis = next_centroid - current_centroid
        else:
            height_axis = np.array([0, 0, 1])
        
        height_axis = height_axis / np.linalg.norm(height_axis)
        
        # Calculate horizontal axis
        y_axis = np.array([0, 1, 0])
        horizontal_axis = np.cross(height_axis, y_axis)
        if np.linalg.norm(horizontal_axis) < 0.1:
            z_axis = np.array([0, 0, 1])
            horizontal_axis = np.cross(height_axis, z_axis)
        horizontal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)
        
        return height_axis, horizontal_axis

    def _mm_to_voxels(self, mm_values):
        """Convert a tuple/list of (dx_mm, dy_mm, dz_mm) to voxel units using spacing."""
        sx, sy, sz = float(self.spacing[0]), float(self.spacing[1]), float(self.spacing[2])
        dx, dy, dz = mm_values
        return max(1, int(round(dx / sx))), max(1, int(round(dy / sy))), max(1, int(round(dz / sz)))

    def _smooth_body_mask_per_vertebra(self, body_mask_bool):
        """Apply 3mm erosion then 2mm dilation to a single vertebra body mask."""
        ex, ey, ez = self._mm_to_voxels((3.0, 3.0, 3.0))
        dx, dy, dz = self._mm_to_voxels((2.0, 2.0, 2.0))

        # Build simple cubic structuring elements sized by voxel radii
        se_erode = np.ones((2*ex+1, 2*ey+1, 2*ez+1), dtype=bool)
        se_dilate = np.ones((2*dx+1, 2*dy+1, 2*dz+1), dtype=bool)

        eroded = binary_erosion(body_mask_bool, footprint=se_erode)
        # eroded = binary_dilation(eroded, footprint=se_dilate)  # do not dilate to save runtime
        return eroded
    
    def _get_horizontal_axis_2d(self, height_axis):
        height_2d_y, height_2d_z = height_axis[1], height_axis[2]
        y_spacing, z_spacing = self.spacing[1], self.spacing[2]
        
        # Transform to physical coordinates
        height_phys_y = height_2d_y * y_spacing
        height_phys_z = height_2d_z * z_spacing
        
        # Calculate perpendicular in physical space
        horiz_phys_y = -height_phys_z
        horiz_phys_z = height_phys_y
        
        # Transform back to pixel coordinates
        horiz_2d_y = horiz_phys_y / y_spacing
        horiz_2d_z = horiz_phys_z / z_spacing
        
        return np.array([0, horiz_2d_y, horiz_2d_z])
    
    def process_vertebrae(self):
        if not self.vertebrae_labels:
            print("No vertebrae labels found starting with 'vertebrae_'")
            return {}
        
        # Filter vertebrae based on spine_range
        filtered_vertebrae, helper_vertebrae = self._filter_vertebrae_by_range()
        
        if not filtered_vertebrae:
            print("No vertebrae to process after filtering")
            return {}
        
        print(f"Found {len(self.sorted_vertebrae)} vertebrae total")
        if self.spine_range != 'all':
            print(f"Processing {len(filtered_vertebrae) - len(helper_vertebrae)} vertebrae in range '{self.spine_range}' (using {len(helper_vertebrae)} helpers for centroid calculation)")
        else:
            print(f"Processing all {len(filtered_vertebrae)} vertebrae:")
        
        vertebrae_info = {}
        centroids = []
        
        # First pass: calculate centroids
        for i, (label_id, label_name) in enumerate(filtered_vertebrae):
            print(f"  {label_name} (ID: {label_id})")
            
            vertebra_mask = self.verte_data == label_id
            vertebra_body_mask = vertebra_mask & self.body_mask
            
            if not np.any(vertebra_body_mask):
                print(f"Warning: No body mask intersection for {label_name}")
                vertebra_body_mask = vertebra_mask
            
            # Smooth per-vertebra body mask for robust centroid calculation
            smoothed_body_mask = self._smooth_body_mask_per_vertebra(vertebra_body_mask.astype(bool))
            # For optional debug export, accumulate smoothed masks in multilabel volume
            self.smoothed_body_multilabel[smoothed_body_mask] = int(label_id)

            # Prefer centroid from smoothed mask; fallback to original if empty
            centroid = self._calculate_centroid(smoothed_body_mask)
            if centroid is None:
                centroid = self._calculate_centroid(vertebra_body_mask)

            if centroid is not None:
                # Check if this is a helper vertebra (only for centroid calculation)
                if label_name in helper_vertebrae:
                    print(f"  -> Helper vertebra (centroid kept for axis calculation)")
                    # Append centroid to keep indices consistent for axis estimation
                    centroids.append(centroid)
                    # Do not add to vertebrae_info to exclude from analysis/visualization
                # Check if this is S1 (skip measurements but keep centroid for L5 axis calculation)
                elif label_name == 'vertebrae_S1':
                    print(f"Skipping {label_name}: S1 excluded from measurements (centroid kept for L5)")
                    # Append centroid to keep indices consistent for axis estimation
                    centroids.append(centroid)
                    # Do not add to vertebrae_info to exclude from analysis/visualization
                # Check if body segmentation touches the image border
                elif (np.any(vertebra_body_mask[0, :, :]) or
                      np.any(vertebra_body_mask[-1, :, :]) or
                      np.any(vertebra_body_mask[:, 0, :]) or
                      np.any(vertebra_body_mask[:, -1, :]) or
                      np.any(vertebra_body_mask[:, :, 0]) or
                      np.any(vertebra_body_mask[:, :, -1])):
                    print(f"Skipping {label_name}: body mask touches image border")
                    # Append centroid to keep indices consistent for axis estimation
                    centroids.append(centroid)
                    # Do not add to vertebrae_info to exclude from analysis/visualization
                else:
                    centroids.append(centroid)
                    vertebrae_info[label_name] = {
                        'label_id': label_id, 'index': i, 'mask': vertebra_mask,
                        'body_mask': vertebra_body_mask, 'body_mask_smoothed': smoothed_body_mask, 'centroid': centroid,
                        'horizontal_mask_type': self.horizontal_mask_type
                    }
            else:
                print(f"Warning: Empty mask for {label_name}")
                centroids.append(None)
        
        # Second pass: calculate axes and measurements
        valid_names = [name for _, name in filtered_vertebrae if name in vertebrae_info]
        for i, vertebra_name in enumerate(valid_names):
            info = vertebrae_info[vertebra_name]
            height_axis, horizontal_axis = self._calculate_axes(centroids, info['index'])
            
            info['height_axis'] = height_axis
            info['horizontal_axis'] = horizontal_axis

            # Compute vertebra body volume (mm^3)
            voxel_volume_mm3 = float(self.spacing[0] * self.spacing[1] * self.spacing[2])
            body_voxels = int(np.count_nonzero(info['body_mask']))
            info['body_volume_mm3'] = float(body_voxels) * voxel_volume_mm3
            
            # Compute mean and median intensity of vertebra body using eroded mask
            erosion_mm = 4
            struct_elem = get_erosion_struct_elem(self.verte_body_img, erosion_mm)
            # use scipy binary_erosion instead of skimage binary_erosion to be consistent with get_vertebrae_body method
            vert_mask_eroded = scipy_binary_erosion(info['body_mask'].astype(bool), structure=struct_elem)
            
            if np.any(vert_mask_eroded):
                body_intensities = self.ct_data[vert_mask_eroded]
                info['mean_intensity'] = float(np.mean(body_intensities))
                info['median_intensity'] = float(np.median(body_intensities))
            else:
                info['mean_intensity'] = 0.0
                info['median_intensity'] = 0.0
            
            # Determine optimal x_slice by scanning across x with 20% margins and ~5mm steps
            horizontal_axis_2d = self._get_horizontal_axis_2d(height_axis)

            # Find x-range where this vertebra body exists using smoothed per-vertebra mask (to avoid spikes)
            body_mask = info['body_mask']
            smoothed_mask = info.get('body_mask_smoothed', body_mask.astype(bool))
            x_presence = np.any(smoothed_mask, axis=(1, 2))
            x_indices = np.where(x_presence)[0]

            best_x_slice = None
            best_measurements = None
            best_spread = -1.0

            if x_indices.size > 0:
                x_min, x_max = int(x_indices[0]), int(x_indices[-1])
                length = x_max - x_min + 1
                if length > 0:
                    # Discard margins at both ends to avoid edge artifacts/spikes
                    margin = int(round(SCAN_MARGIN_RATIO * length))
                    scan_start = min(self.ct_data.shape[0] - 1, max(0, x_min + margin))
                    scan_end = min(self.ct_data.shape[0] - 1, max(0, x_max - margin))

                    if scan_end >= scan_start:
                        # Step in ~5mm increments along x
                        step_slices = max(1, int(round(5.0 / self.spacing[0])))
                        for x in range(scan_start, scan_end + 1, step_slices):
                            body_mask_slice = body_mask[x, :, :]
                            smoothed_mask_slice = smoothed_mask[x, :, :] if smoothed_mask is not None else None
                            horiz_slice = smoothed_mask_slice if self.horizontal_mask_type == 'smoothed' else None
                            measurements = measure_vertebra_heights(
                                body_mask_slice, info['centroid'][1], info['centroid'][2],
                                height_axis, horizontal_axis_2d, self.spacing,
                                mask_horizontal=horiz_slice)

                            heights = [m['height_mm'] for m in measurements.values()]
                            # Require all three to be valid to avoid boundary artifacts
                            if all(h > 0 for h in heights):
                                spread = max(heights) - min(heights)
                                if spread > best_spread:
                                    best_spread = spread
                                    best_x_slice = x
                                    best_measurements = measurements

            # Fallback to centroid slice if no valid candidate found
            if best_x_slice is None:
                fallback_x = max(0, min(int(round(info['centroid'][0])), self.ct_data.shape[0] - 1))
                body_mask_slice = info['body_mask'][fallback_x, :, :]
                smoothed_mask_slice = info.get('body_mask_smoothed', None)
                smoothed_mask_slice = smoothed_mask_slice[fallback_x, :, :] if smoothed_mask_slice is not None else None
                horiz_slice = smoothed_mask_slice if self.horizontal_mask_type == 'smoothed' else None
                best_measurements = measure_vertebra_heights(
                    body_mask_slice, info['centroid'][1], info['centroid'][2],
                    height_axis, horizontal_axis_2d, self.spacing,
                    mask_horizontal=horiz_slice)
                best_x_slice = fallback_x

            info['x_slice'] = int(best_x_slice)
            info['height_measurements'] = best_measurements
        
        return vertebrae_info


def get_verte_height(ct_img, verte_img, verte_label_map, verte_body_img, 
                     verte_body_label_map, preview_file, combined_preview_file, 
                     file_out, debug, smoothed_body_out=None, horizontal_mask_type='smoothed', spine_range='all'):
    """Analyze vertebrae height and create visualization"""
    processor = VertebraeProcessor(ct_img, verte_img, verte_label_map, verte_body_img, verte_body_label_map, horizontal_mask_type=horizontal_mask_type, spine_range=spine_range)
    vertebrae_info = processor.process_vertebrae()
    # Optionally save smoothed per-vertebra body mask for debugging
    if smoothed_body_out:
        save_smoothed_body_mask_nifti(processor, smoothed_body_out, ct_img)
    
    if not vertebrae_info:
        return
    
    vertebrae_names = list(vertebrae_info.keys())
    
    # Create visualizations
    if preview_file:
        create_vertebrae_visualization(ct_img, processor.ct_data, vertebrae_info, vertebrae_names, preview_file, debug)
    
    if combined_preview_file:
        create_combined_spine_visualization(ct_img, processor.ct_data, vertebrae_info, vertebrae_names, combined_preview_file, debug, spine_range)
    
    # Save results
    if file_out:
        save_results(vertebrae_info, file_out)


def save_results(vertebrae_info, file_out):
    """Save measurement results to JSON file"""
    results = {}
    lws_diffs = []
    
    for name, info in vertebrae_info.items():
        if 'height_measurements' not in info:
            continue
            
        # Add height measurements
        for pos, measurement in info['height_measurements'].items():
            medical_term = POSITION_MAPPING.get(pos, pos)
            key = f"{name}_height_{medical_term}_mm"
            results[key] = round(measurement['height_mm'], 2)
        
        # Add height difference
        fracture_info = detect_potential_fracture(info['height_measurements'])
        height_diff = fracture_info['height_difference_percent'] / 100.0
        results[f"{name}_height_difference"] = round(height_diff, 3)
        
        # Collect L1-L5 differences for LWS_max
        if any(name.endswith(f'_L{i}') for i in range(1, 6)):
            lws_diffs.append(height_diff)

        # Add body volume in mL (1 mL = 1000 mm^3)
        body_volume_mm3 = float(info.get('body_volume_mm3', 0.0))
        body_volume_ml = body_volume_mm3 / 1000.0
        results[f"{name}_volume_ml"] = round(body_volume_ml, 2)
        
        # Add mean and median intensity
        mean_intensity = float(info.get('mean_intensity', 0.0))
        median_intensity = float(info.get('median_intensity', 0.0))
        results[f"{name}_intensity"] = round(mean_intensity, 2)
        results[f"{name}_intensity_median"] = round(median_intensity, 2)
    
    # Add LWS_max
    results["LWS_max"] = round(max(lws_diffs), 3) if lws_diffs else None
    
    with open(file_out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {file_out}")
    if lws_diffs:
        print(f"LWS_max (L1-L5 maximum height difference): {results['LWS_max']:.3f}")


def measure_vertebra_heights(mask, center_y, center_z, height_axis, horizontal_axis, spacing, mask_horizontal=None):
    """Measure vertebra height at 3 points along the horizontal axis

    mask: 2D boolean array used for height intersections
    mask_horizontal: optional 2D boolean array used for horizontal axis intersections.
                     If None, falls back to "mask".
    """
    height_2d = height_axis[1:3]
    horiz_2d = horizontal_axis[1:3]
    
    # Use smoothed outline for horizontal axis endpoints when provided
    horiz_mask = mask if mask_horizontal is None else mask_horizontal
    horiz_start, horiz_end = find_axis_intersections(horiz_mask, center_y, center_z, *horiz_2d)
    
    if horiz_start is None or horiz_end is None:
        return {name: _empty_measurement() for name in POSITION_NAMES}
    
    measurements = {}
    for pos, name in zip(MEASUREMENT_POSITIONS, POSITION_NAMES):
        # Calculate measurement point
        measure_y = horiz_start[0] + pos * (horiz_end[0] - horiz_start[0])
        measure_z = horiz_start[1] + pos * (horiz_end[1] - horiz_start[1])
        
        # Find height at this point (use original mask for vertical intersections)
        height_start, height_end = find_axis_intersections(mask, measure_y, measure_z, *height_2d)
        
        if height_start and height_end:
            # Calculate distances
            dy_mm = (height_end[0] - height_start[0]) * spacing[1]
            dz_mm = (height_end[1] - height_start[1]) * spacing[2]
            height_mm = np.sqrt(dy_mm**2 + dz_mm**2)
            height_pixels = np.sqrt(sum((height_end[i] - height_start[i])**2 for i in range(2)))
            
            measurements[name] = {
                'height_mm': height_mm, 'height_pixels': height_pixels,
                'measure_point': (measure_y, measure_z),
                'height_start': height_start, 'height_end': height_end
            }
        else:
            measurements[name] = {
                **_empty_measurement(), 'measure_point': (measure_y, measure_z)
            }
    
    return measurements


def _empty_measurement():
    """Return empty measurement structure"""
    return {'height_mm': 0, 'height_pixels': 0, 'measure_point': None, 
            'height_start': None, 'height_end': None}


def detect_potential_fracture(height_measurements):
    """Detect potential vertebral fracture based on height ratio criteria"""
    heights = [m['height_mm'] for m in height_measurements.values() if m['height_mm'] > 0]
    
    if len(heights) < 2:
        return {'is_fracture': False, 'height_difference_percent': 0}
    
    min_height, max_height = min(heights), max(heights)
    if max_height == 0:
        return {'is_fracture': False, 'height_difference_percent': 0}
    
    ratio = min_height / max_height
    height_diff_percent = round((1 - ratio) * 100)
    is_fracture = ratio < FRACTURE_THRESHOLD
    
    return {'is_fracture': is_fracture, 'height_difference_percent': height_diff_percent}


def find_axis_intersections(mask, center_y, center_z, axis_y, axis_z):
    """Find intersection points where an axis line meets the mask boundary"""
    if not np.any(mask):
        return None, None
    
    # Normalize axis direction
    axis_length = np.sqrt(axis_y**2 + axis_z**2)
    if axis_length == 0:
        return None, None
    
    axis_y_norm, axis_z_norm = axis_y / axis_length, axis_z / axis_length
    max_distance = max(mask.shape) * 1.5
    
    def is_inside_mask(y, z):
        if not (0 <= y < mask.shape[0] and 0 <= z < mask.shape[1]):
            return False
        
        # Bilinear interpolation for sub-pixel accuracy
        y_int, z_int = int(y), int(z)
        y_frac, z_frac = y - y_int, z - z_int
        
        val, count = 0, 0
        for dy in [0, 1]:
            for dz in [0, 1]:
                ny, nz = y_int + dy, z_int + dz
                if 0 <= ny < mask.shape[0] and 0 <= nz < mask.shape[1]:
                    weight = (1 - abs(dy - y_frac)) * (1 - abs(dz - z_frac))
                    val += mask[ny, nz] * weight
                    count += weight
        
        return val > 0.5 if count > 0 else False
    
    def find_intersection(direction):
        """Find intersection in given direction (1 or -1)"""
        for step in np.arange(STEP_SIZE, max_distance, STEP_SIZE):
            y = center_y + direction * axis_y_norm * step
            z = center_z + direction * axis_z_norm * step
            
            if not is_inside_mask(y, z):
                # Back up to find exact boundary
                for backstep in np.arange(step - STEP_SIZE, step, STEP_SIZE/4):
                    by = center_y + direction * axis_y_norm * backstep
                    bz = center_z + direction * axis_z_norm * backstep
                    if is_inside_mask(by, bz):
                        return (by, bz)
                break
        return None
    
    return find_intersection(-1), find_intersection(1)


class VisualizationHelper:
    """Helper class for creating vertebrae visualizations"""
    
    def __init__(self, ct_img, ct_data):
        self.ct_img = ct_img
        self.ct_data = ct_data
        self.spacing = ct_img.header.get_zooms()
    
    def _setup_axes_array(self, n_vertebrae):
        """Setup matplotlib axes array for multiple subplots"""
        cols = min(4, n_vertebrae)
        rows = (n_vertebrae + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if n_vertebrae == 1:
            return fig, [axes]
        elif rows == 1:
            return fig, [axes] if cols == 1 else axes
        else:
            return fig, axes.flatten()
    
    def _get_crop_bounds(self, centroid, crop_size):
        """Calculate crop boundaries around centroid"""
        y_center, z_center = int(round(centroid[1])), int(round(centroid[2]))
        
        y_min = max(0, y_center - crop_size // 2)
        y_max = min(self.ct_data.shape[1], y_center + crop_size // 2)
        z_min = max(0, z_center - crop_size // 2)
        z_max = min(self.ct_data.shape[2], z_center + crop_size // 2)
        
        return y_min, y_max, z_min, z_max
    
    def _crop_data(self, ct_slice, mask_slice, body_mask_slice, bounds):
        """Crop all data arrays to specified bounds"""
        y_min, y_max, z_min, z_max = bounds
        return (
            ct_slice[y_min:y_max, z_min:z_max],
            mask_slice[y_min:y_max, z_min:z_max],
            body_mask_slice[y_min:y_max, z_min:z_max]
        )
    
    def _add_measurements_to_plot(self, ax, info, crop_bounds, crop_data):
        """Add measurement lines and text to plot"""
        y_min, y_max, z_min, z_max = crop_bounds
        ct_crop, mask_crop, body_mask_crop = crop_data
        
        if not np.any(body_mask_crop):
            return
        
        measurements = info['height_measurements']
        fracture_info = detect_potential_fracture(measurements)
        
        for pos in POSITION_NAMES:
            measurement = measurements[pos]
            if not (measurement['height_start'] and measurement['height_end']):
                continue
            
            # Convert coordinates to crop space
            start = (measurement['height_start'][0] - y_min, measurement['height_start'][1] - z_min)
            end = (measurement['height_end'][0] - y_min, measurement['height_end'][1] - z_min)
            
            # Check if within crop bounds
            if not all(0 <= coord < size for coord, size in 
                      [(start[0], ct_crop.shape[0]), (start[1], ct_crop.shape[1]),
                       (end[0], ct_crop.shape[0]), (end[1], ct_crop.shape[1])]):
                continue
            
            # Draw measurement line
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color='red', linewidth=2, alpha=0.8)
            
            # Add text
            mid_y, max_z = (start[0] + end[0]) / 2, max(start[1], end[1])
            height_mm = measurement['height_mm']
            
            if fracture_info['is_fracture']:
                text_color = 'orange'
                height_text = f'{height_mm:.1f} (-{fracture_info["height_difference_percent"]}%)'
            else:
                text_color = 'red'
                height_text = f'{height_mm:.1f}'
            
            ax.text(mid_y, max_z + 3, height_text, color=text_color, fontsize=12, 
                   fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def _add_debug_visualization(self, ax, info, vertebrae_info, vertebrae_names, vertebra_index, crop_bounds, crop_data):
        """Add debug visualization elements (centroids, axes)"""
        y_min, y_max, z_min, z_max = crop_bounds
        ct_crop, mask_crop, body_mask_crop = crop_data
        
        # Get centroid in crop coordinates
        centroid = info['centroid']
        centroid_crop_y = centroid[1] - y_min
        centroid_crop_z = centroid[2] - z_min
        
        # Show centroid
        ax.plot(centroid_crop_y, centroid_crop_z, 'bo', markersize=8, label='Centroid' if vertebra_index == 0 else '')
        
        # Show neighboring vertebrae centroids
        if vertebra_index > 0:
            prev_name = vertebrae_names[vertebra_index - 1]
            prev_centroid = vertebrae_info[prev_name]['centroid']
            prev_y = prev_centroid[1] - y_min
            prev_z = prev_centroid[2] - z_min
            if (0 <= prev_y < ct_crop.shape[0] and 0 <= prev_z < ct_crop.shape[1]):
                ax.plot(prev_y, prev_z, 'mo', markersize=6, alpha=0.8, 
                       label='Prev centroid' if vertebra_index == 0 else '')
        
        if vertebra_index < len(vertebrae_names) - 1:
            next_name = vertebrae_names[vertebra_index + 1]
            next_centroid = vertebrae_info[next_name]['centroid']
            next_y = next_centroid[1] - y_min
            next_z = next_centroid[2] - z_min
            if (0 <= next_y < ct_crop.shape[0] and 0 <= next_z < ct_crop.shape[1]):
                ax.plot(next_y, next_z, 'yo', markersize=6, alpha=0.8,
                       label='Next centroid' if vertebra_index == 0 else '')
        
        # Show height and horizontal axes
        height_axis = info['height_axis']
        
        # Find intersection points for height axis
        height_start, height_end = find_axis_intersections(
            body_mask_crop, centroid_crop_y, centroid_crop_z, 
            height_axis[1], height_axis[2])
        
        # Calculate horizontal axis for display
        y_spacing, z_spacing = self.spacing[1], self.spacing[2]
        height_2d_y, height_2d_z = height_axis[1], height_axis[2]
        
        # Transform to physical coordinates for perpendicular calculation
        height_phys_y = height_2d_y * y_spacing
        height_phys_z = height_2d_z * z_spacing
        
        # Calculate perpendicular in physical space
        horiz_phys_y = -height_phys_z
        horiz_phys_z = height_phys_y
        
        # Transform back to pixel coordinates
        horiz_2d_y = horiz_phys_y / y_spacing
        horiz_2d_z = horiz_phys_z / z_spacing
        
        # Find intersection points for horizontal axis using selected mask type
        smoothed_mask_vol = vertebrae_info[vertebrae_names[vertebra_index]].get('body_mask_smoothed', None)
        use_smoothed = vertebrae_info[vertebrae_names[vertebra_index]].get('horizontal_mask_type', 'smoothed') == 'smoothed'
        if use_smoothed and smoothed_mask_vol is not None:
            smoothed_mask_slice = smoothed_mask_vol[info['x_slice'], :, :]
            smoothed_mask_crop = smoothed_mask_slice[y_min:y_max, z_min:z_max]
            horiz_mask_for_vis = smoothed_mask_crop
        else:
            horiz_mask_for_vis = body_mask_crop
        horiz_start, horiz_end = find_axis_intersections(
            horiz_mask_for_vis, centroid_crop_y, centroid_crop_z, horiz_2d_y, horiz_2d_z)
        
        # Draw axes
        if height_start and height_end:
            ax.plot([height_start[0], height_end[0]], [height_start[1], height_end[1]], 
                   'c-', linewidth=3, label='Height axis' if vertebra_index == 0 else '')
        
        if horiz_start and horiz_end:
            ax.plot([horiz_start[0], horiz_end[0]], [horiz_start[1], horiz_end[1]], 
                   'g-', linewidth=3, label='Horizontal axis' if vertebra_index == 0 else '')
        
        # Add legend to first subplot only
        if vertebra_index == 0:
            ax.legend(fontsize=8)


def create_vertebrae_visualization(ct_img, ct_data, vertebrae_info, vertebrae_names, preview_file, debug):
    """Create PNG visualization showing vertebrae in sagittal view"""
    n_vertebrae = len(vertebrae_names)
    if n_vertebrae == 0:
        print("No vertebrae to visualize")
        return
    
    helper = VisualizationHelper(ct_img, ct_data)
    fig, axes = helper._setup_axes_array(n_vertebrae)
    
    # Calculate crop size
    avg_spacing = np.mean(helper.spacing)
    crop_size = int(round(CROP_SIZE_MM / avg_spacing))
    aspect_ratio = helper.spacing[2] / helper.spacing[1]
    
    for i, vertebra_name in enumerate(vertebrae_names[:len(axes)]):
        info = vertebrae_info[vertebra_name]
        x_slice = info['x_slice']
        
        # Get slices
        ct_slice = ct_data[x_slice, :, :]
        mask_slice = info['mask'][x_slice, :, :]
        body_mask_slice = info['body_mask'][x_slice, :, :]
        
        # Crop data
        crop_bounds = helper._get_crop_bounds(info['centroid'], crop_size)
        crop_data = helper._crop_data(ct_slice, mask_slice, body_mask_slice, crop_bounds)
        ct_crop, mask_crop, body_mask_crop = crop_data
        
        # Setup plot
        ax = axes[i]
        ax.imshow(ct_crop.T, cmap='gray', origin='lower', aspect=aspect_ratio, vmin=HU_WINDOW[0], vmax=HU_WINDOW[1])
        
        # Add contours
        if np.any(body_mask_crop):
            contours = measure.find_contours(body_mask_crop, 0.5)
            for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], 'green', linewidth=1.5, alpha=0.8)
        
        # Add measurements
        helper._add_measurements_to_plot(ax, info, crop_bounds, crop_data)
        
        # Add debug visualization
        if debug:
            helper._add_debug_visualization(ax, info, vertebrae_info, vertebrae_names, i, crop_bounds, crop_data)
        
        # Setup axes
        x_slice_display = ct_data.shape[0] - 1 - x_slice
        ax.set_title(f'{vertebra_name}\nSagittal slice x={x_slice_display}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_xaxis()
    
    # Hide unused subplots
    for i in range(n_vertebrae, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(preview_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {preview_file}")
    print(f"Created {n_vertebrae} vertebrae visualizations")


def save_smoothed_body_mask_nifti(processor, out_path: str, reference_img):
    """Save smoothed per-vertebra body mask as a multilabel NIfTI.
    Uses the reference image affine/header for spatial metadata.
    """
    data = processor.smoothed_body_multilabel.astype(np.uint16)
    img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(img, out_path)


def _calculate_combined_bounds(ct_data, vertebrae_info, vertebrae_names, spacing):
    """Calculate bounding box for combined visualization"""
    all_y_coords, all_z_coords = [], []
    
    for vertebra_name in vertebrae_names:
        info = vertebrae_info[vertebra_name]
        mask = info['mask']
        x_slice = info['x_slice']
        vertebra_mask_slice = mask[x_slice, :, :]
        
        if np.any(vertebra_mask_slice):
            coords = np.where(vertebra_mask_slice)
            if len(coords[0]) > 0:
                all_y_coords.extend(coords[0])
                all_z_coords.extend(coords[1])
    
    if not all_y_coords:
        return None
    
    # Calculate bounds with padding
    y_padding = int(round(PADDING_MM / spacing[1]))
    z_padding = int(round(PADDING_MM / spacing[2]))
    
    return (
        max(0, min(all_y_coords) - y_padding),
        min(ct_data.shape[1], max(all_y_coords) + y_padding),
        max(0, min(all_z_coords) - z_padding),
        min(ct_data.shape[2], max(all_z_coords) + z_padding)
    )


def _create_combined_image(ct_data, vertebrae_info, vertebrae_names, bounds):
    """Create combined image with all vertebrae"""
    y_min, y_max, z_min, z_max = bounds
    crop_y_size, crop_z_size = y_max - y_min, z_max - z_min
    
    combined_image = np.full((crop_y_size, crop_z_size), -1000.0, dtype=np.float32)
    filled_mask = np.zeros((crop_y_size, crop_z_size), dtype=bool)
    
    for vertebra_name in vertebrae_names:
        info = vertebrae_info[vertebra_name]
        x_slice = info['x_slice']
        
        ct_slice = ct_data[x_slice, :, :]
        vertebra_mask_slice = info['mask'][x_slice, :, :]
        
        # Crop to region of interest
        ct_slice_crop = ct_slice[y_min:y_max, z_min:z_max]
        vertebra_mask_crop = vertebra_mask_slice[y_min:y_max, z_min:z_max]
        
        # Add pixels that haven't been filled yet
        vertebra_pixels = vertebra_mask_crop & ~filled_mask
        combined_image[vertebra_pixels] = ct_slice_crop[vertebra_pixels]
        filled_mask[vertebra_pixels] = True
    
    return combined_image, filled_mask


def _add_combined_annotations(ax, ct_data, vertebrae_info, vertebrae_names, bounds, spacing, spine_range='all'):
    """Add vertebrae annotations to combined plot"""
    y_min, y_max, z_min, z_max = bounds
    
    # Use larger font size for lumbar-only view
    fontsize = 13 if spine_range == 'lumbar' else 9
    
    for vertebra_name in vertebrae_names:
        info = vertebrae_info[vertebra_name]
        centroid = info['centroid']
        x_slice = info['x_slice']
        
        body_mask_slice = info['body_mask'][x_slice, :, :]
        body_mask_slice_crop = body_mask_slice[y_min:y_max, z_min:z_max]
        
        # Add contours
        if np.any(body_mask_slice_crop):
            contours = measure.find_contours(body_mask_slice_crop, 0.5)
            for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], 'green', linewidth=1.5, alpha=0.8)
        
        # Add measurement lines and text
        measurements = info.get('height_measurements', {})
        if measurements:
            vertebra_id = vertebra_name.replace('vertebrae_', '')
            height_values = []
            
            # Draw measurement lines in red
            for pos in POSITION_NAMES:
                measurement = measurements[pos]
                if measurement['height_start'] and measurement['height_end']:
                    # Convert to crop coordinates
                    start = (measurement['height_start'][0] - y_min, measurement['height_start'][1] - z_min)
                    end = (measurement['height_end'][0] - y_min, measurement['height_end'][1] - z_min)
                    
                    # Draw measurement line
                    ax.plot([start[0], end[0]], [start[1], end[1]], 
                           color='red', linewidth=2, alpha=0.8)
                    
                    height_values.append(f"{measurement['height_mm']:.1f}")
                else:
                    height_values.append('0.0')
            
            if height_values:
                heights_text = ' - '.join(reversed(height_values))  # Reverse for display
                fracture_info = detect_potential_fracture(measurements)
                
                x_slice_display = ct_data.shape[0] - 1 - x_slice
                
                # Use stricter 30% threshold for L5 when deciding yellow highlight
                is_fracture_for_display = (
                    # fracture_info['height_difference_percent'] >= 30
                    # if vertebra_id == 'L5' else fracture_info['is_fracture']
                    fracture_info['is_fracture']
                )
                
                # Get median intensity for display
                median_intensity = info.get('median_intensity', 0.0)
                median_intensity_int = int(round(median_intensity))
                
                # Compute body volume in mL for thresholding (not displayed anymore)
                body_volume_mm3 = info.get('body_volume_mm3', 0.0)
                body_volume_ml = body_volume_mm3 / 1000.0
                # Force yellow if volume below threshold, independent of height difference
                force_yellow_due_to_volume = (body_volume_ml < BODY_VOLUME_YELLOW_THRESHOLD_ML)
                
                # Add line break after height difference for lumbar view
                line_break = '\n' if spine_range == 'lumbar' else ' '
                
                if is_fracture_for_display or force_yellow_due_to_volume:
                    text = f"{vertebra_id}: {heights_text} (-{fracture_info['height_difference_percent']}%){line_break}(x: {x_slice_display}, HU: {median_intensity_int})"
                    text_color = 'yellow'
                else:
                    text = f"{vertebra_id}: {heights_text}{line_break}(x: {x_slice_display}, HU: {median_intensity_int})"
                    text_color = 'white'
                
                # Position text
                text_y_crop = (centroid[1] - y_min) + (TEXT_OFFSET_MM / spacing[1])
                text_z_crop = centroid[2] - z_min
                
                ax.text(text_y_crop, text_z_crop, text, color=text_color, fontsize=fontsize, 
                       fontweight='bold', ha='left', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))


def create_combined_spine_visualization(ct_img, ct_data, vertebrae_info, vertebrae_names, preview_file, debug, spine_range='all'):
    """Create PNG visualization showing all vertebrae combined in one sagittal view"""
    if not vertebrae_names:
        print("No vertebrae to visualize in combined view")
        return
    
    spacing = ct_img.header.get_zooms()
    bounds = _calculate_combined_bounds(ct_data, vertebrae_info, vertebrae_names, spacing)
    
    if bounds is None:
        print("No vertebrae coordinates found for cropping")
        return
    
    y_min, y_max, z_min, z_max = bounds
    print(f"Cropping combined view to region: Y[{y_min}:{y_max}] Z[{z_min}:{z_max}] ({y_max-y_min}x{z_max-z_min} pixels)")
    print(f"Creating combined spine visualization with {len(vertebrae_names)} vertebrae")
    
    # Create combined image
    combined_image, filled_mask = _create_combined_image(ct_data, vertebrae_info, vertebrae_names, bounds)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 12), facecolor='black')
    ax.set_facecolor('black')
    
    aspect_ratio = spacing[2] / spacing[1]
    ax.imshow(combined_image.T, cmap='gray', origin='lower', aspect=aspect_ratio, vmin=HU_WINDOW[0], vmax=HU_WINDOW[1])
    
    # Add annotations
    _add_combined_annotations(ax, ct_data, vertebrae_info, vertebrae_names, bounds, spacing, spine_range)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(preview_file, dpi=600, bbox_inches='tight', facecolor='black')
    plt.close()
    
    filled_pixels = np.sum(filled_mask)
    total_pixels = (y_max - y_min) * (z_max - z_min)
    print(f"Combined spine visualization saved to {preview_file}")
    print(f"Filled {filled_pixels}/{total_pixels} pixels ({100*filled_pixels/total_pixels:.1f}%)")
    print(f"Cropped from original size {ct_data.shape[1]}x{ct_data.shape[2]} to {y_max-y_min}x{z_max-z_min}")


"""
Test:

# Faster (15mm)
cd ~/Downloads/nnunet_test/osteoporosis_fractures/29268064
python ~/dev/jakob_scripts/osteoporosis_fracture/measure_verte_height.py -i ct.nii.gz -v totalseg.nii.gz -b verte_body.nii.gz -p preview.png -c preview_combined.png -o results.json --debug -s smoothed_body.nii.gz

# Slower (higher res; anisotropic)
cd ~/Downloads/nnunet_test/osteoporosis_fractures/29276218_2
python ~/dev/jakob_scripts/osteoporosis_fracture/measure_verte_height.py -i ct.nii.gz -v totalseg.nii.gz -b verte_body.nii.gz -p preview.png -c preview_combined.png -o results.json --debug -s smoothed_body.nii.gz
"""
if __name__ == "__main__":
    """
    Measure vertebral height
    """
    st = time.time()

    parser = argparse.ArgumentParser(usage="Measure vertebral height.")
    parser.add_argument('-i', '--ct_file', type=Path, required=True, help="CT file.")
    parser.add_argument('-v', '--vertebrae_seg_file', type=Path, required=True, help="vertebrae segmentation file.")
    parser.add_argument('-b', '--vertebrae_body_seg_file', type=Path, required=True, help="vertebrae body segmentation file.")
    parser.add_argument('-p', '--preview_file', type=Path, help="png image showing individual vertebrae measurements")
    parser.add_argument('-c', '--combined_preview_file', type=Path, help="png image showing combined spine view")
    parser.add_argument('-o', '--file_out', type=Path, help="json output")
    parser.add_argument('-s', '--smoothed_body_out', type=Path, help="Optional path to save smoothed vertebrae body mask NIfTI for debugging")
    parser.add_argument('--horizontal-mask', choices=['smoothed', 'raw'], default='raw',
                        help="Mask to use for horizontal axis endpoints: 'smoothed' (default) or 'raw'")
    parser.add_argument('--spine_range', choices=['all', 'thoracic_lumbar', 'lumbar', 'l1_l4'], default='all',
                        help="Limit analysis to a specific vertebral range")
    parser.add_argument('-d', '--debug', action='store_true', help="Print debug information")
    args = parser.parse_args()

    ct_img = nib.load(args.ct_file)
    ct_img = nib.as_closest_canonical(ct_img)
    # label_map: a dictionary {label_id : label_name}
    # this contains binary segmentations of each vertebrae (e.g. vertebrae_C1,...,vertebrae_L5)
    verte_img, verte_label_map = load_multilabel_nifti(args.vertebrae_seg_file)
    verte_img = nib.as_closest_canonical(verte_img)
    # this contains binary segmentations of the vertebrae body of all vertebrae in one class (vertebrae_body)
    verte_body_img, verte_body_label_map = load_multilabel_nifti(args.vertebrae_body_seg_file)
    verte_body_img = nib.as_closest_canonical(verte_body_img)
    
    get_verte_height(ct_img, verte_img, verte_label_map, verte_body_img, 
                    verte_body_label_map, args.preview_file, 
                    args.combined_preview_file, args.file_out, args.debug,
                    str(args.smoothed_body_out) if args.smoothed_body_out else None,
                    horizontal_mask_type=args.horizontal_mask,
                    spine_range=args.spine_range)
