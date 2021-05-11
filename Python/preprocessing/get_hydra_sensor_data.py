#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for Hydra Sensordata Report that can handle missing camera data

@author: sm5911
@date: 29/04/2021

"""

import argparse
from pathlib import Path

from tierpsytools.hydra.read_imgstore_extradata import ExtraDataReader

EXAMPLE_ROOT_DIR = '/Volumes/hermes$/KeioScreen_96WP/RawVideos/20210427'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hydra Sensordata Report')
    parser.add_argument('--root_dir', help="RawVideo root directory for hydra sensordata report",
                        default=EXAMPLE_ROOT_DIR, type=str)
    args = parser.parse_args()  

    root_dir = Path(args.root_dir)

    files2process = list(root_dir.rglob('metadata.yaml'))
    for fname in files2process:
        try:
            edr = ExtraDataReader(fname)
            foo = edr.get_extra_data()
        except:
            print(f'failed: {fname}')
