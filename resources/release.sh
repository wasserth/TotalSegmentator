#!/bin/bash
set -e  # Exit on error

# Info: Have to run from within the resources directory otherwise paths incorrect
#
# Run first: tests/test_locally.py
#
# use nnunetv2 env
#
# Usage: ./release.sh -> will ask for new version number

# go to root of package
cd ..


echo "Reminder: First run tests/test_locally.py"

echo "Reminder: First update CHANGELOG.md"

# Function to update the version in setup.py
update_version() {
  old_version=$(grep -oP "(?<=version=')[^']*(?=')" setup.py)
  echo "Current version: ${old_version}"
  read -p "Enter new version: " new_version

  # Replace old version with new version in setup.py
  sed -i "s/version='${old_version}'/version='${new_version}'/g" setup.py
}

# Step 1: Update version in setup.py
update_version

# Step 2: Commit the version update and tag it
git add setup.py
git commit -m "Bump version to ${new_version}"
git tag "v${new_version}"

# Step 3: Push commits and tags to GitHub
git push origin master
git push origin "v${new_version}"

# Step 4: Publish the package to PyPI
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*

# Step 5: Build and Push Docker Image
#   (random error on my local machine; have to run on server)
# docker build -t wasserth/totalsegmentator:${new_version} .
# docker push wasserth/totalsegmentator:${new_version}
echo "Build and upload docker container manually"

echo "Release process completed."
