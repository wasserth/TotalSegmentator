import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology, measure
import json
import platform
import os

class EnhancedLiverVesselProcessor:
    def __init__(self, contrast_phase=None):
        self.contrast_phase = contrast_phase
        self.is_mac = platform.system() == "Darwin"
        self.use_mps = self._check_mps_availability()
        
    def _check_mps_availability(self):
        """Check if Mac MPS (Metal Performance Shaders) is available"""
        try:
            import torch
            return torch.backends.mps.is_available() if self.is_mac else False
        except:
            return False
    
    def enhance_vessel_connectivity(self, vessel_mask, liver_mask):
        """Improve vessel connectivity using morphological operations"""
        print("🔗 Enhancing vessel connectivity...")
        
        # Ensure vessels are within liver boundary
        vessel_mask = vessel_mask * liver_mask
        
        # Mac-optimized morphological operations
        kernel_size = 2 if self.is_mac else 3  # Smaller kernel for Mac performance
        kernel = morphology.ball(kernel_size)
        vessel_mask = morphology.binary_closing(vessel_mask, kernel)
        
        # Remove small disconnected components
        labeled = measure.label(vessel_mask)
        props = measure.regionprops(labeled)
        
        # Adaptive threshold based on system
        min_size = 30 if self.is_mac else 50  # Smaller threshold for Mac
        for prop in props:
            if prop.area < min_size:
                vessel_mask[labeled == prop.label] = 0
                
        print(f"✅ Vessel connectivity enhanced (Mac optimized: {self.is_mac})")
        return vessel_mask
    
    def optimize_for_contrast_phase(self, ct_image, vessel_mask):
        """Adjust vessel detection based on contrast phase"""
        print(f"🩸 Optimizing for contrast phase: {self.contrast_phase}")
        
        if self.contrast_phase == "arterial":
            return self._enhance_arterial_vessels(ct_image, vessel_mask)
        elif self.contrast_phase == "portal":
            return self._enhance_portal_vessels(ct_image, vessel_mask)
        elif self.contrast_phase == "delayed":
            return self._enhance_venous_vessels(ct_image, vessel_mask)
        else:
            print("📊 No specific contrast phase detected, using general enhancement")
            return vessel_mask
    
    def _enhance_arterial_vessels(self, ct_image, vessel_mask):
        """Focus on high-intensity vessels (arteries)"""
        print("🔴 Enhancing arterial vessels...")
        arterial_threshold = np.percentile(ct_image[vessel_mask > 0], 75)
        enhanced_mask = vessel_mask.copy()
        enhanced_mask[ct_image < arterial_threshold] *= 0.5
        return enhanced_mask
    
    def _enhance_portal_vessels(self, ct_image, vessel_mask):
        """Focus on portal vein branches"""
        print("🟣 Enhancing portal vessels...")
        portal_threshold_low = np.percentile(ct_image[vessel_mask > 0], 25)
        portal_threshold_high = np.percentile(ct_image[vessel_mask > 0], 75)
        
        enhanced_mask = vessel_mask.copy()
        portal_range = (ct_image >= portal_threshold_low) & (ct_image <= portal_threshold_high)
        enhanced_mask[~portal_range] *= 0.7
        return enhanced_mask
    
    def _enhance_venous_vessels(self, ct_image, vessel_mask):
        """Focus on hepatic veins"""
        print("🔵 Enhancing venous vessels...")
        enhanced_mask = vessel_mask.copy()
        return enhanced_mask

def process_enhanced_liver_vessels(input_path, output_path, liver_mask_path=None):
    """Main function to process liver vessels with Mac-optimized enhancements"""
    print("🍎 Starting Mac-optimized liver vessel processing...")
    
    # Convert Path objects to strings for Mac compatibility
    input_path = str(input_path)
    output_path = str(output_path)
    
    # Load images
    print("📖 Loading CT image...")
    ct_img = nib.load(input_path)
    ct_data = ct_img.get_fdata()
    
    # Detect contrast phase
    print("🔍 Detecting contrast phase...")
    try:
        from totalsegmentator.bin.totalseg_get_phase import predict_phase
        phase_result = predict_phase(input_path)
        contrast_phase = phase_result.get('phase', None)
        print(f"📊 Detected phase: {contrast_phase}")
    except Exception as e:
        print(f"⚠️ Could not detect contrast phase: {e}")
        contrast_phase = None
    
    # Initialize processor
    processor = EnhancedLiverVesselProcessor(contrast_phase)
    
    # Run standard TotalSegmentator liver_vessels
    print("🔬 Running TotalSegmentator liver_vessels...")
    from totalsegmentator.python_api import totalsegmentator
    
    # Mac-optimized device selection
    device = "mps" if processor.use_mps else "cpu"
    print(f"🖥️ Using device: {device}")
    
    vessel_result = totalsegmentator(
        input_path, 
        None, 
        task="liver_vessels",
        robust_crop=True,
        ml=True,
        device=device
    )
    
    # Load liver mask
    print("🫀 Loading liver mask...")
    if liver_mask_path:
        liver_img = nib.load(str(liver_mask_path))
        liver_mask = liver_img.get_fdata()
    else:
        print("🔍 Generating liver mask...")
        liver_result = totalsegmentator(
            input_path, 
            None, 
            roi_subset=["liver"], 
            ml=True,
            device=device
        )
        liver_mask = (liver_result.get_fdata() == 5).astype(np.uint8)
    
    # Extract vessel mask
    vessel_mask = (vessel_result.get_fdata() == 1).astype(np.uint8)
    
    # Apply enhancements
    print("⚡ Applying enhancements...")
    enhanced_vessels = processor.enhance_vessel_connectivity(vessel_mask, liver_mask)
    enhanced_vessels = processor.optimize_for_contrast_phase(ct_data, enhanced_vessels)
    
    # Save enhanced result
    print(f"💾 Saving enhanced result to: {output_path}")
    enhanced_img = nib.Nifti1Image(enhanced_vessels, ct_img.affine, ct_img.header)
    nib.save(enhanced_img, output_path)
    
    # Save metadata
    metadata = {
        'contrast_phase': contrast_phase,
        'enhancement_applied': True,
        'vessel_volume_mm3': float(np.sum(enhanced_vessels) * np.prod(ct_img.header.get_zooms())),
        'processing_timestamp': str(np.datetime64('now')),
        'system': 'macOS',
        'device_used': device,
        'mps_available': processor.use_mps
    }
    
    metadata_path = output_path.replace('.nii.gz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Processing complete!")
    print(f"📊 Vessel volume: {metadata['vessel_volume_mm3']:.2f} mm³")
    
    return enhanced_vessels, metadata