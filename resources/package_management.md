# Helpful commands to manage the package

## Update version and publish new version to pypi

Increase version number in code and commit
- setup.py
- Changelog

```
git push
git tag -a v1.5 -m "version 1.5"
git push origin --tags
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
```
