from setuptools import setup, find_packages


setup(name='TotalSegmentator',
        version='2.1.0',
        description='Robust segmentation of 104 classes in CT images.',
        long_description="See Readme.md on github for more details.",
        url='https://github.com/wasserth/TotalSegmentator',
        author='Jakob Wasserthal',
        author_email='jakob.wasserthal@usb.ch',
        python_requires='>=3.9',
        license='Apache 2.0',
        packages=find_packages(),
        install_requires=[
            'torch>=1.10.2',
            'numpy',
            'psutil',
            'SimpleITK',
            'nibabel>=2.3.0',
            'tqdm>=4.45.0',
            'p_tqdm',
            'xvfbwrapper',
            'fury',
            'nnunetv2>=2.2.1',
            'requests==2.27.1;python_version<"3.10"',
            'requests;python_version>="3.10"',
            'rt_utils',
            'dicom2nifti',
            'pyarrow'
        ],
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
                'totalseg_combine_masks=totalsegmentator.bin.totalseg_combine_masks:main',
                'crop_to_body=totalsegmentator.bin.crop_to_body:main',
                'totalseg_import_weights=totalsegmentator.bin.totalseg_import_weights:main',
                'totalseg_download_weights=totalsegmentator.bin.totalseg_download_weights:main',
                'totalseg_setup_manually=totalsegmentator.bin.totalseg_setup_manually:main',
                'totalseg_set_license=totalsegmentator.bin.totalseg_set_license:main'
            ],
        },
    )
