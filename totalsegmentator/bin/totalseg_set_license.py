#!/usr/bin/env python
import sys
from pathlib import Path
import argparse

from totalsegmentator.config import setup_totalseg, set_license_number


def main():
    """
    Set your totalsegmentator license number

    Usage:
    totalseg_set_license -l aca_12345678910
    """
    parser = argparse.ArgumentParser(description="Set license.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-l", "--license_number", type=str, help="TotalSegmentator license number.",
                        required=True)

    args = parser.parse_args()

    if not args.license_number.startswith("aca_"):
        raise ValueError("license number must start with 'aca_' or 'com_' ")
    if len(args.license_number) != 18:
        raise ValueError("license number must have exactly 18 characters.")

    setup_totalseg()  # create config file if not exists
    set_license_number(args.license_number)

    print("License has been sucessfully saved.")


if __name__ == "__main__":
    main()
