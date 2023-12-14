# Copyright (C) 2023 Cyprien Quéméneur
# For the full license, please refer to the LICENSE file in the root directory of this project.
# For the full copyright notices, please refer to the NOTICE file in the root directory of this project.

import argparse
import os
from PIL import Image
import random
import shutil
from tqdm import tqdm
from datasets_utils import create_directories, get_distribution_dataframe, convert_bbox

KITTI_TRAIN_SIZE = 7481
DEFAULT_CLASS_MAP = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7
}


def get_iid_splits(nclients: int, val_frac: float) -> dict:
    """Return a dictionary to store IID and balanced mapping of KITTI data split."""
    random.seed(0)
    client_frac = (1 - val_frac) / nclients
    indices = list(range(KITTI_TRAIN_SIZE))
    client_split_size = int(KITTI_TRAIN_SIZE * client_frac)
    splits = {}
    # Create the client splits
    for k in range(1, nclients + 1):
        client_data = random.sample(indices, client_split_size)
        for index in client_data:
            splits[index] = f'client{k}'
            indices.remove(index)
    # Create the server split
    for index in indices:
        splits[index] = 'server'
    return splits


def process_kitti(img_path: str, label_path: str, target_path: str, data: str, class_map: dict, nclients: int,
                  val_frac: float) -> None:
    """Convert KITTI annotations and split the data among the server and clients."""
    print('Converting annotations and splitting data...')
    create_directories(target_path, nclients)
    splits = get_iid_splits(nclients, val_frac)
    objects_distribution = get_distribution_dataframe(data, nclients)
    # Iterate over KITTI training labels
    for fname in tqdm(os.listdir(label_path)):
        # Create target file
        destination = splits[int(fname[:-4])]
        objects_distribution.loc['Samples', destination] += 1
        with (open(f'{target_path}/{destination}/labels/{fname}', 'w') as target_file):
            # Open KITTI training label
            with open(f'{label_path}/{fname}', 'r') as label_file:
                # Open KITTI corresponding image and extract image width and height
                with open(f'{img_path}/{fname[:-3]}png', 'rb') as img_file:
                    img = Image.open(img_file)
                    img_width, img_height = img.size
                # Copy the image to its destination without deleting the original file
                shutil.copyfile(f'{img_path}/{fname[:-3]}png', f'{target_path}/{destination}/images/{fname[:-3]}png')
                # Iterate over KITTI training label lines
                for line in label_file.readlines():
                    line = line.split()
                    obj_type, _, _, _, bbox_left, bbox_top, bbox_right, bbox_bottom, *_ = line
                    # Skip line with DontCare type
                    if obj_type == 'DontCare':
                        continue
                    # Convert KITTI training label line to YOLO format [class_id, x, y, w, h]
                    class_id = class_map[obj_type]
                    x, y, w, h = convert_bbox(
                        bbox_left=float(bbox_left),
                        bbox_top=float(bbox_top),
                        bbox_right=float(bbox_right),
                        bbox_bottom=float(bbox_bottom),
                        img_width=img_width,
                        img_height=img_height
                    )
                    # Write processed label line to target file
                    target_file.write(f'{class_id} {x} {y} {w} {h}\n')
                    # Update object distribution
                    objects_distribution.loc[obj_type, destination] += 1
    # Save objects distribution
    objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--img-path', type=str, default='datasets/data_object_image_2/training/image_2')
    args.add_argument('--label-path', type=str, default='datasets/data_object_label_2/training/label_2')
    args.add_argument('--target-path', type=str, default='datasets/kitti', help='path to target directory')
    args.add_argument('--data', type=str, help='path to data yaml file')
    args.add_argument('--class-map', type=dict, default=None, help='map between annotations, should match yaml file')
    args.add_argument('--nclients', type=int, default=5, help='number of clients in federated experiment')
    args.add_argument('--val-frac', type=float, default=0.25, help='fraction of data held by the server for validation')
    args = args.parse_args()
    class_map = DEFAULT_CLASS_MAP if args.class_map is None else args.class_map
    process_kitti(args.img_path, args.label_path, args.target_path, args.data, class_map, args.nclients, args.val_frac)
