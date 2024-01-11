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