def rgb_to_cielab_dicom(rgb):
    """
    Convert RGB (0-255) to CIE Lab color space for DICOM SEG.
    DICOM uses unsigned 16-bit integers (0-65535) to store CIE Lab values.
    
    Args:
        rgb: tuple of (R, G, B) values in range 0-255
        
    Returns:
        tuple of (L, a, b) values encoded as unsigned integers (0-65535) for DICOM
    """
    # Normalize RGB to 0-1
    r, g, b = [x / 255.0 for x in rgb]
    
    # Convert RGB to XYZ (using sRGB color space)
    # Apply gamma correction
    def gamma_correct(c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    
    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)
    
    # Convert to XYZ using sRGB matrix
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # Normalize by D65 white point
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883
    
    # Convert XYZ to Lab
    def f(t):
        delta = 6/29
        if t > delta**3:
            return t**(1/3)
        else:
            return t / (3 * delta**2) + 4/29
    
    L = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b_lab = 200 * (f(y) - f(z))
    
    # Encode for DICOM (unsigned 16-bit integers)
    # L*: 0-100 scaled to 0-65535
    L_dicom = int(round(L * 65535 / 100))
    # a* and b*: approximately -128 to +127, shifted and scaled to 0-65535
    # Using the formula: (value + 128) * 65535 / 255
    a_dicom = int(round((a + 128) * 65535 / 255))
    b_dicom = int(round((b_lab + 128) * 65535 / 255))
    
    # Clamp values to valid range
    L_dicom = max(0, min(65535, L_dicom))
    a_dicom = max(0, min(65535, a_dicom))
    b_dicom = max(0, min(65535, b_dicom))
    
    return (L_dicom, a_dicom, b_dicom)


def generate_random_color():
    """
    Generate a random vibrant color in RGB.
    
    Returns:
        tuple of (R, G, B) values in range 0-255
    """
    import random
    # Generate colors with high saturation for better visibility
    hue = random.random()
    saturation = 0.7 + random.random() * 0.3  # 0.7 to 1.0
    value = 0.7 + random.random() * 0.3  # 0.7 to 1.0
    
    # Convert HSV to RGB
    h_i = int(hue * 6)
    f = hue * 6 - h_i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)
    
    if h_i == 0:
        r, g, b = value, t, p
    elif h_i == 1:
        r, g, b = q, value, p
    elif h_i == 2:
        r, g, b = p, value, t
    elif h_i == 3:
        r, g, b = p, q, value
    elif h_i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q
    
    return (int(r * 255), int(g * 255), int(b * 255))


def load_snomed_mapping():
    """Load SNOMED CT codes mapping from CSV file."""
    import csv
    from pathlib import Path
    
    csv_path = Path(__file__).parent / "resources" / "totalsegmentator_snomed_mapping.csv"
    
    snomed_map = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            structure = row['Structure']
            snomed_map[structure] = {
                'property_category': {
                    'scheme': row['SegmentedPropertyCategoryCodeSequence.CodingSchemeDesignator'],
                    'value': row['SegmentedPropertyCategoryCodeSequence.CodeValue'],
                    'meaning': row['SegmentedPropertyCategoryCodeSequence.CodeMeaning']
                },
                'property_type': {
                    'scheme': row['SegmentedPropertyTypeCodeSequence.CodingSchemeDesignator'],
                    'value': row['SegmentedPropertyTypeCodeSequence.CodeValue'],
                    'meaning': row['SegmentedPropertyTypeCodeSequence.CodeMeaning']
                },
                'property_modifier': {
                    'scheme': row['SegmentedPropertyTypeModifierCodeSequence.CodingSchemeDesignator'],
                    'value': row['SegmentedPropertyTypeModifierCodeSequence.CodeValue'],
                    'meaning': row['SegmentedPropertyTypeModifierCodeSequence.CodeMeaning']
                } if row['SegmentedPropertyTypeModifierCodeSequence.CodeValue'] else None,
                'anatomic_region': {
                    'scheme': row['AnatomicRegionSequence.CodingSchemeDesignator'],
                    'value': row['AnatomicRegionSequence.CodeValue'],
                    'meaning': row['AnatomicRegionSequence.CodeMeaning']
                } if row['AnatomicRegionSequence.CodeValue'] else None,
                'anatomic_modifier': {
                    'scheme': row['AnatomicRegionModifierSequence.CodingSchemeDesignator'],
                    'value': row['AnatomicRegionModifierSequence.CodeValue'],
                    'meaning': row['AnatomicRegionModifierSequence.CodeMeaning']
                } if row['AnatomicRegionModifierSequence.CodeValue'] else None
            }
    
    return snomed_map
