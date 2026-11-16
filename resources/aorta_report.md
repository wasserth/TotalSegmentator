# Aorta report

`totalseg_aorta_report` creates a quantitative report of the thoracic and abdominal aorta from a
CT image. It places 11 standardized landmarks along the aortic centerline, measures the vessel at
each landmark, summarizes the main aortic sections, and renders the results as JSON and an RGB
NIfTI report. It can also generate a curved planar reformation (CPR).

The report is a research prototype and is not intended for clinical diagnosis.
For research use, cite the
[TotalSegmentator Aorta Report paper](https://arxiv.org/abs/2611.17968) (see [Citation](#citation)).

Overview of the aorta report

## Requirements

Running the segmentation models requires a TotalSegmentator license. A license is free for
non-commercial use and can be requested at
[backend.totalsegmentator.com/license-academic](https://backend.totalsegmentator.com/license-academic/).
Configure it once with:

```bash
totalseg_set_license -l aca_XXXXXXXX
```

The report renderer requires `wkhtmltopdf` and a virtual X server. On Ubuntu/Debian:

```bash
sudo apt-get install wkhtmltopdf xvfb
```

Contrast-phase detection requires `xgboost`:

```bash
pip install xgboost
```



## Running the complete pipeline

Use `--run_models` to create all required masks from the CT:

```bash
totalseg_aorta_report \
  -i ct.nii.gz \
  -o aorta_report.nii.gz \
  -j aorta_report.json \
  -l aorta_report.log \
  -c aorta_cpr.png \
  --run_models
```

Patient metadata can optionally be supplied with `-n nodeinfo.json`. Without it, anonymous default  
metadata is used. The metadata is displayed in the rendered report and included in the JSON.

## Segmentation pipeline

1. A fast `total` task segments the aorta and heart to define a crop.
2. The CT is cropped with a 20 mm margin around the aorta and heart.
3. The cropped CT is resampled to 0.8 mm isotropic spacing.
4. The normal `total` task produces the aorta, branch vessels, iliac arteries, and T12 masks.
5. The `aortic_dissection` task produces true- and false-lumen masks.
6. The `aorta_annulus` task produces the annulus and sinotubular junction.
7. The `renal_arteries` task produces the celiac trunk, superior mesenteric
  artery, and renal arteries.
8. `totalseg_get_phase` classifies the contrast phase.

## Landmark definitions

The report follows 11 positions from the aortic root to the abdominal bifurcation:

1. **Annulus** — the annulus mask at the cardiac end of the centerline.
2. **Sinuses of Valsalva** — the maximum-diameter position between the annulus and sinotubular
  junction, excluding a 10 mm margin at both ends. The midpoint is used when this interval is too
   short.
3. **Sinotubular junction** — the centerline position closest to the sinotubular-junction mask.
4. **Mid ascending aorta** — halfway between landmarks 3 and 5.
5. **Distal ascending aorta** — 5 mm before the brachiocephalic-trunk origin.
6. **Mid aortic arch** — halfway between landmarks 5 and 7.
7. **Proximal descending aorta** — 5 mm distal to the left subclavian-artery origin.
8. **Mid descending aorta** — halfway between landmarks 7 and 9.
9. **Descending aorta at T12** — the centerline position closest to the center of T12. If T12 is
  detected below the celiac landmark, a robust fallback places it 10 mm above the celiac position.
10. **Abdominal aorta at the celiac artery** — the upper end of the celiac-artery intersection.
11. **Abdominal aorta near the bifurcation** — 20 mm above the distal centerline endpoint, using the
  right iliac artery to establish that the bifurcation is present.

Landmarks 1-3 depend on the annulus and/or sinotubular-junction masks; landmarks 4-8 additionally
depend on the brachiocephalic trunk, left subclavian artery, or T12; landmarks 9-11 depend on T12,
the celiac artery, or the right iliac artery respectively.

## Landmark measurements

For each available landmark, the code constructs a plane perpendicular to the local centerline.
The annulus plane uses the orientation of the annulus mask itself. The plane is intersected with the
aorta, rotated into a two-dimensional image, closed, and hole-filled.

The following values are stored:

- total, true-lumen, and false-lumen cross-sectional area in cm²;
- maximum diameter and a near-perpendicular diameter in cm;
- corresponding true- and false-lumen diameters in cm;
- diameter endpoints in both voxel and world coordinates.

`--erosion` erodes the measurement mask by one 0.8 mm voxel before constructing the plane. This can
reduce slight over-segmentation but changes all diameter and area measurements.

True/false-lumen measurements are disabled when the lumen masks are absent, when
`--skip_dissection` is set, or when the scan is classified as native/non-contrast. In that case the
complete aorta is treated as true lumen and the false lumen is empty.

## Aortic section measurements

The landmarks define five sections:

- ascending aorta: landmarks 1 to 5;
- aortic arch: landmarks 5 to 7;
- descending aorta: landmarks 7 to 9;
- abdominal aorta: landmarks 9 to 11;
- complete aorta: landmarks 1 to 11.

For each available section, the report computes centerline length in cm, maximum diameter in cm,  
total volume in ml, true-lumen volume in ml, and false-lumen volume in ml. If either boundary  
landmark is unavailable, all values for that section are `null`.

## CPR and rendered report

When `--cpr`  is requested, the CT and masks are sampled on planes transported along the centerline. The CPR overview contains sagittal and coronal straightened views, vessel contours, landmark positions, and diameter profiles for the complete aorta and both lumens.

## JSON structure

The top-level JSON contains:

- `landmarks`: entries `1` through `11` with names and available measurements;
- `section_stats`: measurements for the five aortic sections;
- `metadata`: normalized patient/study metadata and the report version.

Unavailable measurements are represented as `null`. Diameter endpoint coordinates can vary by a  
small amount between runs when several boundary-point pairs have the same diameter.

## Limitations

- Report quality depends directly on aorta, branch-vessel, annulus, and landmark segmentation
quality.
- Missing anatomy or a restricted field of view can make landmarks and complete sections
unavailable.
- The centerline assumes one dominant connected aortic path; severe discontinuities can produce a
short or incorrect path.
- Native scans do not use the dissection model output for lumen measurements.
- CPR and 3D rendering are computationally and memory intensive.
- The implementation currently supports local model execution only.
- Input validation currently returns without an error status, so automation should verify that all
requested output files were created.

## Citation

If you use the aorta report please cite our [paper](https://arxiv.org/abs/2611.17968):

```text
Wasserthal, J., Hinck, D., Segeroth, M., Cyriac, J., Berdajs, D., Miazza, J., Pradella, M., 2026. TotalSegmentator Aorta Report: Automated Aortic Diameter Measurement at 11 Standardized Landmarks on CT Images. arXiv:2611.17968. https://doi.org/10.48550/arXiv.2611.17968
```

Please also cite the [TotalSegmentator paper](https://pubs.rsna.org/doi/10.1148/ryai.230024), since the aorta report is built on TotalSegmentator segmentations.

## License

The aorta report is [free](https://backend.totalsegmentator.com/license-academic/) for non-commercial usage. A license is required for commercial usage. 