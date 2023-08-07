# Mapping of classes which are not part of 6mm model (total_fast) to a class which is part of the model.
# This is needed to crop to this class when using --roi_subset.
map_to_total_fast = {

    # requires 6mm body model instead of 6mm total
    "patella": "body_extremities",
    "tibia": "body_extremities",
    "fibula": "body_extremities",
    "tarsal": "body_extremities",
    "metatarsal": "body_extremities",
    "phalanges_feet": "body_extremities",
    "ulna": "body_extremities",
    "radius": "body_extremities",
    "carpal": "body_extremities",
    "metacarpal": "body_extremities",
    "phalanges_hand": "body_extremities",

    # requires 6mm body model instead of 6mm total
    "subcutaneous_fat": "body",
    "skeletal_muscle": "body",
    "torso_fat": "body",

    "heart_myocardium": "heart",
    "heart_atrium_left": "heart", 
    "heart_ventricle_left": "heart", 
    "heart_atrium_right": "heart", 
    "heart_ventricle_right": "heart", 
    "pulmonary_artery": "heart",

    "face": "skull"
}