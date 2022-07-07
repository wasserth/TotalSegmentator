from setuptools import setup, find_packages

setup(name='TotalSegmentator',
        version='0.1',
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
            'nibabel>=2.3.0',
            'tqdm',
            'p_tqdm',
            'xvfbwrapper',
            'fury',
            'batchgenerators==0.21',
            'nnunet @ git+https://github.com/wasserth/nnUNet_cust@working_2022_03_18#egg=nnUNet'
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
            'bin/TotalSegmentator', 'bin/totalseg_combine_masks'
        ]
    )
