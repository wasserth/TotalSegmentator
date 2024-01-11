# Mapping of classes of commercial models (which are not part of 6mm model) to a class which is part of
# 6mm the model. This is needed to crop to this class when using --roi_subset.
map_to_total = {

    # Mapping to class from body_model does not work with current implementation!
    # for tissue: simply use option: -bs
    # for bones: no solution at the moment; also use -bs

    # "patella": "body_extremities",
    # "tibia": "body_extremities",
    # "fibula": "body_extremities",
    # "tarsal": "body_extremities",
    # "metatarsal": "body_extremities",
    # "phalanges_feet": "body_extremities",
    # "ulna": "body_extremities",
    # "radius": "body_extremities",
    # "carpal": "body_extremities",
    # "metacarpal": "body_extremities",
    # "phalanges_hand": "body_extremities",

    # "subcutaneous_fat": "body",
    # "skeletal_muscle": "body",
    # "torso_fat": "body",

    "heart_myocardium": "heart",
    "heart_atrium_left": "heart",
    "heart_ventricle_left": "heart",
    "heart_atrium_right": "heart",
    "heart_ventricle_right": "heart",
    "pulmonary_artery": "heart",

    "face": "skull"
}