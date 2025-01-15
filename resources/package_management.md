# Helpful commands to manage the package


## Run pre-commit checks
```
pre-commit run --all-file
```


## Update version and publish new version to pypi

Increase version number in code and commit
- setup.py
- Changelog

```
comm "increase version to 1.5.1"
git push
git tag -a v1.5.1 -m "version 1.5.1"
git push origin --tags
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
```

INFO: Now all done by `release.sh`


## Release new weights
* Run `./resources/prepare_weights_for_release.sh DATASET_ID [DATASET_ID2 ...]`
* Or do it manually:
    * `cd /mnt/nvme/data/multiseg/weights_upload/totalsegmentator_v2`
    * `cp -r $nnUNet_results/Dataset527_breasts_1559subj .`
    * `python ~/dev/TotalSegmentator/resources/anonymise_nnunet_pkl_v2.py Dataset527_breasts_1559subj/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres`
    * `zip -r Dataset527_breasts_1559subj.zip Dataset527_breasts_1559subj`
* Click on "Draft a new release" on github
* Create new tag ending with -weights and upload weights
