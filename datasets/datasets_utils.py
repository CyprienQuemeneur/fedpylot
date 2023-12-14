# Copyright (C) 2023 Cyprien Quéméneur
# For the full license, please refer to the LICENSE file in the root directory of this project.
# For the full copyright notices, please refer to the NOTICE file in the root directory of this project.

import os
import pandas as pd
import yaml


def create_directories(target_path: str, nclients: int) -> None:
    """Create directory structure for the data."""
    # Create target directory
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # Create server subdirectory
    for subdict in ['', '/images', '/labels']:
        if not os.path.exists(f'{target_path}/server{subdict}'):
            os.makedirs(f'{target_path}/server{subdict}')
    # Create client subdirectories
    for k in range(1, nclients + 1):
        for subdict in ['', '/images', '/labels']:
            if not os.path.exists(f'{target_path}/client{k}{subdict}'):
                os.makedirs(f'{target_path}/client{k}{subdict}')


def get_distribution_dataframe(data: str, nclients: int) -> pd.DataFrame:
    """Create a dataframe to store the distribution of the annotations between the nodes."""
    columns = ['server']
    with open(data, 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    for k in range(1, nclients + 1):
        columns.append(f'client{k}')
    objects_distribution = pd.DataFrame(columns=columns, index=['Samples'] + data_dict['names'])
    objects_distribution.fillna(0, inplace=True)
    return objects_distribution


def convert_bbox(bbox_left: float, bbox_top: float, bbox_right: float, bbox_bottom: float, img_width: int,
                 img_height: int) -> tuple[float, float, float, float]:
    """Convert bounding box annotations to YOLO format."""
    x = (bbox_left + bbox_right) / 2 / img_width
    y = (bbox_top + bbox_bottom) / 2 / img_height
    w = (bbox_right - bbox_left) / img_width
    h = (bbox_bottom - bbox_top) / img_height
    return x, y, w, h
