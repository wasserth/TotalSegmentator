import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 10):
    # Specify the fixed version for Python < 3.10. Because using the latest
    # requests would also install the latest urllib3 which does not work
    # properly on python < 3.10.
    requests_version = '==2.27.1'  #requires: urllib3>=1.21.1,<1.27 
    # 2.27.1 somehow not available in dockerfile
else:
    requests_version = ''  # No fixed version for Python 3.10 and higher


setup(name='TotalSegmentator',
        version='2.0.4',
        description='Robust segmentation of 104 classes in CT images.',
        long_description="See Readme.md on github for more details.",
        url='https://github.com/wasserth/TotalSegmentator',
        author='Jakob Wasserthal',
        author_email='jakob.wasserthal@usb.ch',
        python_requires='>=3.5',
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
            'nnunetv2==2.2',
            f'requests{requests_version}',
            'rt_utils',
            'dicom2nifti'
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
