/**
 * Organ Color Configuration
 * Extracted from totalseg_material.py for consistency with Blender visualization
 */

// Material definitions with exact anatomical colors
export const MATERIALS = {
  "Bone": [0.509338, 0.448805, 0.390992],
  "Muscle": [0.458575, 0.114023, 0.099804],
  "Liver": [0.359082, 0.052501, 0.044477],
  "Stomach": [0.483567, 0.277414, 0.269021],
  "Artery": [0.675526, 0.020398, 0.041993],
  "Vein": [0.071473, 0.01412, 0.37347],
  "Kidney": [0.359082, 0.084555, 0.06085],
  "Adrenal": [0.799999, 0.254006, 0.03054],
  "Pancreas": [0.450415, 0.259994, 0.102502],
  "GB": [0.12796, 0.16291, 0.069646],
  "Heart": [0.675526, 0.020398, 0.041993],
  "Portal": [0.047439, 0.046528, 0.434352],
  "Lung": [0.475151, 0.316953, 0.299059],
  "Thyroid": [0.37347, 0.24654, 0.067987],
  "Bladder": [0.591379, 0.383078, 0.371987],
  "Spleen": [0.110568, 0.021969, 0.025012],
  "Prostate": [0.366235, 0.160072, 0.063098],
  "Colon": [0.403241, 0.212665, 0.103747],
};

// Direct organ name to material mapping
export const ORGAN_TO_MATERIAL = {
  // Bone
  "skull": "Bone",
  "costal_cartilages": "Bone",
  "sternum": "Bone",
  "clavicle_left": "Bone",
  "clavicle_right": "Bone",
  "scapula_left": "Bone",
  "scapula_right": "Bone",
  "humerus_left": "Bone",
  "humerus_right": "Bone",
  "vertebrae_c3": "Bone",
  "vertebrae_c4": "Bone",
  "vertebrae_c5": "Bone",
  "vertebrae_c6": "Bone",
  "vertebrae_c7": "Bone",
  "vertebrae_t1": "Bone",
  "vertebrae_t2": "Bone",
  "vertebrae_t3": "Bone",
  "vertebrae_t4": "Bone",
  "vertebrae_t5": "Bone",
  "vertebrae_t6": "Bone",
  "vertebrae_t7": "Bone",
  "vertebrae_t8": "Bone",
  "vertebrae_t9": "Bone",
  "vertebrae_t10": "Bone",
  "vertebrae_t11": "Bone",
  "vertebrae_t12": "Bone",
  "vertebrae_l1": "Bone",
  "vertebrae_l2": "Bone",
  "vertebrae_l3": "Bone",
  "vertebrae_l4": "Bone",
  "vertebrae_l5": "Bone",
  "vertebrae_s1": "Bone",
  "sacrum": "Bone",
  "rib_left_1": "Bone",
  "rib_left_2": "Bone",
  "rib_left_3": "Bone",
  "rib_left_4": "Bone",
  "rib_left_5": "Bone",
  "rib_left_6": "Bone",
  "rib_left_7": "Bone",
  "rib_left_8": "Bone",
  "rib_left_9": "Bone",
  "rib_left_10": "Bone",
  "rib_left_11": "Bone",
  "rib_left_12": "Bone",
  "rib_right_1": "Bone",
  "rib_right_2": "Bone",
  "rib_right_3": "Bone",
  "rib_right_4": "Bone",
  "rib_right_5": "Bone",
  "rib_right_6": "Bone",
  "rib_right_7": "Bone",
  "rib_right_8": "Bone",
  "rib_right_9": "Bone",
  "rib_right_10": "Bone",
  "rib_right_11": "Bone",
  "rib_right_12": "Bone",
  "hip_left": "Bone",
  "hip_right": "Bone",
  "femur_left": "Bone",
  "femur_right": "Bone",
  "spinal_cord": "Bone",
  
  // Muscle
  "iliopsoas_left": "Muscle",
  "iliopsoas_right": "Muscle",
  "autochthon_left": "Muscle",
  "autochthon_right": "Muscle",
  "gluteus_maximus_left": "Muscle",
  "gluteus_medius_left": "Muscle",
  "gluteus_minimus_left": "Muscle",
  "gluteus_maximus_right": "Muscle",
  "gluteus_medius_right": "Muscle",
  "gluteus_minimus_right": "Muscle",
  
  // Thoracic
  "heart": "Heart",
  "atrial_appendage_left": "Heart",
  "pulmonary_vein": "Vein",
  "thyroid_gland": "Thyroid",
  "lung_upper_lobe_left": "Lung",
  "lung_lower_lobe_left": "Lung",
  "lung_upper_lobe_right": "Lung",
  "lung_middle_lobe_right": "Lung",
  "lung_lower_lobe_right": "Lung",
  
  // Abdominal
  "urinary_bladder": "Bladder",
  "prostate": "Prostate",
  "colon": "Colon",
  "duodenum": "Stomach",
  "esophagus": "Stomach",
  "gallbladder": "GB",
  "adrenal_gland_left": "Adrenal",
  "adrenal_gland_right": "Adrenal",
  "kidney_left": "Kidney",
  "kidney_right": "Kidney",
  "liver": "Liver",
  "pancreas": "Pancreas",
  "small_bowel": "Stomach",
  "spleen": "Spleen",
  "stomach": "Stomach",
  
  // Vessel
  "aorta": "Artery",
  "brachiocephalic_trunk": "Artery",
  "subclavian_artery_left": "Artery",
  "subclavian_artery_right": "Artery",
  "common_carotid_artery_left": "Artery",
  "common_carotid_artery_right": "Artery",
  "iliac_artery_left": "Artery",
  "iliac_artery_right": "Artery",
  "superior_vena_cava": "Vein",
  "inferior_vena_cava": "Vein",
  "portal_vein_and_splenic_vein": "Portal",
  "brachiocephalic_vein_left": "Vein",
  "brachiocephalic_vein_right": "Vein",
  "iliac_vena_left": "Vein",
  "iliac_vena_right": "Vein",
  "blood_vessel": "Vein",
};

// Category definitions for UI grouping
export const CATEGORIES = {
  "Bone": {
    name: "Skeletal System",
    icon: "🦴",
    organs: Object.keys(ORGAN_TO_MATERIAL).filter(k => ORGAN_TO_MATERIAL[k] === "Bone")
  },
  "Muscle": {
    name: "Muscular System",
    icon: "💪",
    organs: Object.keys(ORGAN_TO_MATERIAL).filter(k => ORGAN_TO_MATERIAL[k] === "Muscle")
  },
  "Vessel": {
    name: "Circulatory System",
    icon: "❤️",
    organs: Object.keys(ORGAN_TO_MATERIAL).filter(k => 
      ["Artery", "Vein", "Portal", "Heart"].includes(ORGAN_TO_MATERIAL[k])
    )
  },
  "Thoracic": {
    name: "Respiratory System",
    icon: "🫁",
    organs: Object.keys(ORGAN_TO_MATERIAL).filter(k => 
      ["Lung", "Thyroid"].includes(ORGAN_TO_MATERIAL[k])
    )
  },
  "Abdominal": {
    name: "Digestive System",
    icon: "🫀",
    organs: Object.keys(ORGAN_TO_MATERIAL).filter(k => 
      ["Liver", "Stomach", "Pancreas", "GB", "Colon", "Kidney", "Spleen", "Bladder", "Prostate", "Adrenal"].includes(ORGAN_TO_MATERIAL[k])
    )
  }
};

// Get color for organ name
export function getColorForOrgan(organName) {
  // Normalize name (remove .stl, lowercase, replace hyphens)
  const normalized = organName.replace('.stl', '').toLowerCase().replace(/-/g, '_');
  
  // Look up material
  const material = ORGAN_TO_MATERIAL[normalized];
  
  if (material && MATERIALS[material]) {
    return MATERIALS[material];
  }
  
  // Default gray color
  return [0.7, 0.7, 0.7];
}

// Get category for organ
export function getCategoryForOrgan(organName) {
  const normalized = organName.replace('.stl', '').toLowerCase().replace(/-/g, '_');
  const material = ORGAN_TO_MATERIAL[normalized];
  
  for (const [category, info] of Object.entries(CATEGORIES)) {
    if (info.organs.includes(normalized)) {
      return category;
    }
  }
  
  return "Other";
}