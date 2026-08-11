VESSELS_TO_PLOT = (
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "common_carotid_artery_left",
    "renal_arteries",
    "celiac_trunk",
    "superior_mesenteric_artery",
    "iliac_artery_left",
    "iliac_artery_right",
)

STRUCTURE_DEFINITIONS = {
    "brachio": ("brachiocephalic_trunk", "totalseg"),
    "subclavian": ("subclavian_artery_left", "totalseg"),
    "celiac": ("celiac_trunk", "totalseg"),
    "T12": ("vertebrae_T12", "totalseg"),
    "sinotub_junc": ("sinotubular_junction", "details"),
    "iliac": ("iliac_artery_right", "totalseg"),
}

LANDMARK_DEPENDENCIES = {
    1: ("annulus",),
    2: ("annulus", "sinotub_junc"),
    3: ("sinotub_junc",),
    4: ("sinotub_junc", "brachio"),
    5: ("brachio",),
    6: ("brachio", "subclavian"),
    7: ("subclavian",),
    8: ("subclavian", "T12"),
    9: ("T12",),
    10: ("celiac",),
    11: ("iliac",),
}

LANDMARK_NAMES = {
    1: "annulus",
    2: "sinuses of valsalva",
    3: "sinotub. junc.",
    4: "mid asc. aorta",
    5: "distal asc. aorta",
    6: "mid aortic arch",
    7: "proximal desc. aorta",
    8: "mid desc. aorta",
    9: "desc. aorta (T12)",
    10: "abd. aorta (celiac artery)",
    11: "abd. aorta (bifurcation)",
}

SECTION_LANDMARKS = {
    "aorta_ascending": (1, 5),
    "aorta_arch": (5, 7),
    "aorta_descending": (7, 9),
    "aorta_abdominal": (9, 11),
    "aorta_total": (1, 11),
}

REPORT_PLOT_TYPES = ("preview_3d_rotating_", "preview_tf_lumen_")
REPORT_FRAMES = 24
REPORT_SMOOTHING = 120
