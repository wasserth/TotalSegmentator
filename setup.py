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
        version='1.5.7',
        description='Robust segmentation of 104 classes in CT images.',
        long_description="See Readme.md on github for more details.",
        url='https://github.com/wasserth/TotalSegmentator',
        author='Jakob Wasserthal',
        author_email='jakob.wasserthal@usb.ch',
        python_requires='>=3.5',
        license='Apache 2.0',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'psutil',
            # Any version <2.1.0 because of this issue: 
            # https://github.com/SimpleITK/SimpleITK/issues/1433
            'SimpleITK',
            'nibabel>=2.3.0',
            'tqdm>=4.45.0',
            'p_tqdm',
            'xvfbwrapper',
            'fury',
            'batchgenerators==0.21',
            # This does not work if want to upload to pypi
            # 'nnunet @ git+https://github.com/wasserth/nnUNet_cust@working_2022_03_18#egg=nnUNet'
            'nnunet-customized==1.2',
            f'requests{requests_version}',
            'rt_utils'
        ],
        zip_safe=False,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        scripts=[
            'bin/TotalSegmentator', 'bin/totalseg_combine_masks', 'bin/crop_to_body', 
            'bin/totalseg_import_weights', 'bin/totalseg_download_weights',
            'bin/totalseg_setup_manually'
        ]
    )
