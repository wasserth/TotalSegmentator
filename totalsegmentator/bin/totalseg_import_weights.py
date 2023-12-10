#!/usr/bin/env python
import os
import sys
from pathlib import Path
import argparse
import zipfile

from totalsegmentator.config import get_totalseg_dir, get_weights_dir


def main():
    """
    Import manually downloaded weights (zip file) to the right folder.
    DEPRECATED! This is no longer needed in v2.0.0 and later.
    """
    parser = argparse.ArgumentParser(description="Import manually downloaded weights.",
                                     epilog="Written by Jakob Wasserthal.")

    parser.add_argument("-i", "--weights_file",
                        help="path to the weights zip file",
                        type=lambda p: Path(p).absolute(), required=True)

    args = parser.parse_args()

    config_dir = get_weights_dir()
    config_dir.mkdir(exist_ok=True, parents=True)

    print(f"Extracting file {args.weights_file} to {config_dir}")

    with zipfile.ZipFile(args.weights_file, 'r') as zip_f:
        zip_f.extractall(config_dir)


if __name__ == "__main__":
    main()
