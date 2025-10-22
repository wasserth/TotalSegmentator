import bpy
import re
import sys
import os

# Check for output filename in command line arguments
# Format: blender -b file.blend -P material_fix.py -- output.blend
output_filename = "scene_colored.blend"  # Default name
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]  # Get args after "--"
    if len(argv) >= 1:
        output_filename = argv[0]

# First, create all the materials with exact colors
def create_material(name, color):
    # Remove existing material if present
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    
    # Create new material
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    
    # Set color in Principled BSDF
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs['Base Color'].default_value = color
    
    # Set viewport display color too
    material.diffuse_color = color
    
    print(f"Created material '{name}' with color: {color}")
    return material

# Create all materials with EXACTLY the same colors as in your original script
materials = {
    "Bone": create_material("Bone", (0.509338, 0.448805, 0.390992, 1.0)),
    "Muscle": create_material("Muscle", (0.458575, 0.114023, 0.099804, 1.0)),
    "Liver": create_material("Liver", (0.359082, 0.052501, 0.044477, 1.0)),
    "Stomach": create_material("Stomach", (0.483567, 0.277414, 0.269021, 1.0)),
    "Artery": create_material("Artery", (0.675526, 0.020398, 0.041993, 1.0)),
    "Vein": create_material("Vein", (0.071473, 0.01412, 0.37347, 1.0)),
    "Kidney": create_material("Kidney", (0.359082, 0.084555, 0.06085, 1.0)),
    "Adrenal": create_material("Adrenal", (0.799999, 0.254006, 0.03054, 1.0)),
    "Pancreas": create_material("Pancreas", (0.450415, 0.259994, 0.102502, 1.0)),
    "GB": create_material("GB", (0.12796, 0.16291, 0.069646, 1.0)),
    "Heart": create_material("Heart", (0.675526, 0.020398, 0.041993, 1.0)),
    "Portal": create_material("Portal", (0.047439, 0.046528, 0.434352, 1.0)),
    "Lung": create_material("Lung", (0.475151, 0.316953, 0.299059, 1.0)),
    "Thyroid": create_material("Thyroid", (0.37347, 0.24654, 0.067987, 1.0)),
    "Bladder": create_material("Bladder", (0.591379, 0.383078, 0.371987, 1.0)),
    "Spleen": create_material("Spleen", (0.110568, 0.021969, 0.025012, 1.0)),
    "Prostate": create_material("Prostate", (0.366235, 0.160072, 0.063098, 1.0)),
    "Colon": create_material("Colon", (0.403241, 0.212665, 0.103747, 1.0)),
    # Also add "Adrenall" for the typo in your original script
    "Adrenall": create_material("Adrenall", (0.799999, 0.254006, 0.03054, 1.0)),
}

# Helper function to normalize and compare names
def normalize_name(name):
    name = name.lower()
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name)
    return name

# Map normalized names directly to the exact names from your script
direct_mappings = {
    "skull": "Bone",
    "costal_cartilages": "Bone",
    "sternum": "Bone",
    "left_clavicle": "Bone",
    "right_clavicle": "Bone", 
    "left_scapula": "Bone",
    "right_scapula": "Bone",
    "left_humerus": "Bone",
    "right_humerus": "Bone",
    # Vertebrae
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
    # Ribs
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
    
    # Muscles
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
    
    # Thoracic
    "trachea": "",
    "heart": "Heart",
    "left_atrial_appendage": "Heart",
    "pulmonary_vein": "Vein",
    "thyroid": "Thyroid",
    "lung_upper_lobe_right": "Lung",
    "lung_lower_lobe_left": "Lung",
    "lung_lower_lobe_right": "Lung",
    "lung_middle_lobe_right": "Lung",
    "lung_upper_lobe_left": "Lung",
    
    # Abdominal
    "urinary_bladder": "Bladder",
    "prostate": "Prostate",
    "colon": "Colon",
    "duodenum": "Stomach",
    "esophagus": "Stomach",
    "gallbladder": "GB",
    "adrenal_gland_left": "Adrenall",  # Note the original typo
    "adrenal_gland_right": "Adrenal",
    "kidney_left": "Kidney",
    "kidney_right": "Kidney",
    "liver": "Liver",
    "pancreas": "Pancreas",
    "small_bowel": "Stomach",
    "spleen": "Spleen",
    "stomach": "Stomach",
    
    # Vessel
    "aorta": "Artery",
    "superior_vena_cava": "Vein",
    "inferior_vena_cava": "Vein",
    "brachiocephalic_vein_left": "Vein",
    "brachiocephalic_vein_right": "Vein",
    "subclavian_artery_left": "Artery",
    "subclavian_artery_right": "Artery",
    "brachiocephalic_trunk": "Vein",
    "iliac_artery_left": "Artery",
    "iliac_artery_right": "Artery",
    "carotid_artery_left": "Artery",
    "carotid_artery_right": "Artery",
    "iliac_vena_left": "Vein",
    "iliac_vena_right": "Vein",
    "portal_vein_and_splenic_vein": "Vein",
    "blood_vessel": "Vein",
}

print("\n=== Applying Materials to Objects ===")

# Apply materials to all objects in the scene
applied_count = 0
missing_count = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
        
    # Try direct name mapping first
    material_name = direct_mappings.get(obj.name.lower())
    
    # If no direct match, try pattern matching as fallback
    if not material_name:
        obj_name = normalize_name(obj.name)
        
        # Try to find a matching pattern using regex
        for pattern, mat_name in direct_mappings.items():
            if re.search(pattern.replace("_", "[_ ]?"), obj_name, re.IGNORECASE):
                material_name = mat_name
                break
    
    # If no material or empty material name, skip
    if not material_name:
        print(f"No material mapping for: {obj.name}")
        missing_count += 1
        continue
    
    # Get the material
    material = bpy.data.materials.get(material_name)
    if not material:
        print(f"Material not found: {material_name} for object {obj.name}")
        missing_count += 1
        continue
    
    # Apply material
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    
    applied_count += 1
    print(f"Applied '{material_name}' to '{obj.name}'")

print(f"\nComplete: Applied materials to {applied_count} objects. {missing_count} objects without matching materials.")

# Make sure output directory exists
output_dir = os.path.dirname(os.path.abspath(f"./out/{output_filename}"))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the file with the provided name
output_path = os.path.abspath(f"./out/{output_filename}")
bpy.ops.wm.save_as_mainfile(filepath=output_path)

print(f"File saved to: {output_path}")