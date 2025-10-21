from setuptools import setup, find_packages


setup(name='TotalSegmentator',
        version='2.11.0',
        description='Robust segmentation of 104 classes in CT images.',
        long_description="See Readme.md on github for more details.",
        url='https://github.com/wasserth/TotalSegmentator',
        author='Jakob Wasserthal',
        author_email='jakob.wasserthal@usb.ch',
        python_requires='>=3.9',
        license='Apache 2.0',
        packages=find_packages(),
        package_data={"totalsegmentator": 
            ["resources/totalsegmentator_snomed_mapping.csv",
             "resources/contrast_phase_classifiers_2024_07_19.pkl",
             "resources/modality_classifiers_2025_02_24.json.0",
             "resources/modality_classifiers_2025_02_24.json.1",
             "resources/modality_classifiers_2025_02_24.json.2",
             "resources/modality_classifiers_2025_02_24.json.3",
             "resources/modality_classifiers_2025_02_24.json.4",
             "resources/modality_classifiers_normalized_2025_02_24.json.0",
             "resources/modality_classifiers_normalized_2025_02_24.json.1",
             "resources/modality_classifiers_normalized_2025_02_24.json.2",
             "resources/modality_classifiers_normalized_2025_02_24.json.3",
             "resources/modality_classifiers_normalized_2025_02_24.json.4",
             "resources/ct_brain_atlas_1mm.nii.gz"]
            },
        install_requires=[
            'torch>=2.1.2',  # torch < 2.6.0 required if using nnunetv2 < 2.6.0
            'numpy',
            'SimpleITK',
            'nibabel>=2.3.0',
            'tqdm>=4.45.0',
            'xvfbwrapper',
            'nnunetv2>=2.3.1',
            'requests==2.27.1;python_version<"3.10"',
            'requests;python_version>="3.10"',
            'dicom2nifti',
            'pyarrow',
            'Pillow',
            'pydicom'
        ],
        extras_require={
            'enhanced': [
                'scikit-image',  # For smoothing and mesh processing
                'trimesh',       # For STL export to Blender
                'scipy'          # For advanced image processing
            ]
        },
        zip_safe=False,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        entry_points={
            'console_scripts': [
                'TotalSegmentator=totalsegmentator.bin.TotalSegmentator:main',
                'TotalSegmentatorEnhanced=totalsegmentator.bin.TotalSegmentatorEnhanced:main',
                'TotalSegmentatorImproved=totalsegmentator.bin.TotalSegmentatorImproved:main',
                'totalseg_combine_masks=totalsegmentator.bin.totalseg_combine_masks:main',
                'crop_to_body=totalsegmentator.bin.crop_to_body:main',
                'totalseg_import_weights=totalsegmentator.bin.totalseg_import_weights:main',
                'totalseg_download_weights=totalsegmentator.bin.totalseg_download_weights:main',
                'totalseg_setup_manually=totalsegmentator.bin.totalseg_setup_manually:main',
                'totalseg_set_license=totalsegmentator.bin.totalseg_set_license:main',
                'totalseg_get_phase=totalsegmentator.bin.totalseg_get_phase:main',
                'totalseg_get_modality=totalsegmentator.bin.totalseg_get_modality:main',
                'totalseg_evans_index=totalsegmentator.bin.totalseg_evans_index:main',
                'totalseg_dicom_to_png=totalsegmentator.bin.totalseg_dicom_to_png:main',
                'totalseg_dataset_build=totalsegmentator.bin.totalseg_dataset_build:main',
                'totalseg_blender_import=totalsegmentator.bin.totalseg_blender_import:main',
                'totalseg_blender_slider=totalsegmentator.bin.totalseg_blender_slider:main',
                'totalseg_mpr_widget=totalsegmentator.bin.totalseg_mpr_widget:main'
            ],
        },
    )
