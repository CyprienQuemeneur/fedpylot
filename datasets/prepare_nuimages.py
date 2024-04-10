# Copyright (C) 2024 Cyprien Quéméneur
# FedPylot is released under the GPL-3.0 license, please refer to the LICENSE file in the root directory of the program.
# For the full copyright notices, please refer to the NOTICE file in the root directory of the program.

import argparse
import os
import random
import shutil
from nuimages import NuImages
import pandas as pd
from tqdm import tqdm
from datasets_utils import create_directories, archive_directories, get_distribution_dataframe, convert_bbox

CLASS_MAP10 = {
    'animal': None,
    'human.pedestrian.adult': (0, 'pedestrian'),
    'human.pedestrian.child': (0, 'pedestrian'),
    'human.pedestrian.construction_worker': (0, 'pedestrian'),
    'human.pedestrian.personal_mobility': None,
    'human.pedestrian.police_officer': (0, 'pedestrian'),
    'human.pedestrian.stroller': None,
    'human.pedestrian.wheelchair': None,
    'movable_object.barrier': (1, 'barrier'),
    'movable_object.debris': None,
    'movable_object.pushable_pullable': None,
    'movable_object.trafficcone': (2, 'traffic_cone'),
    'static_object.bicycle_rack': None,
    'vehicle.bicycle': (3, 'bicycle'),
    'vehicle.bus.bendy': (4, 'bus'),
    'vehicle.bus.rigid': (4, 'bus'),
    'vehicle.car': (5, 'car'),
    'vehicle.construction': (6, 'construction_vehicle'),
    'vehicle.emergency.ambulance': None,
    'vehicle.emergency.police': None,
    'vehicle.motorcycle': (7, 'motorcycle'),
    'vehicle.trailer': (8, 'trailer'),
    'vehicle.truck': (9, 'truck'),
    'vehicle.ego': None
}

CLASS_MAP23 = {
    'animal': 0,
    'human.pedestrian.adult': 1,
    'human.pedestrian.child': 2,
    'human.pedestrian.construction_worker': 3,
    'human.pedestrian.personal_mobility': 4,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.stroller': 6,
    'human.pedestrian.wheelchair': 7,
    'movable_object.barrier': 8,
    'movable_object.debris': 9,
    'movable_object.pushable_pullable': 10,
    'movable_object.trafficcone': 11,
    'static_object.bicycle_rack': 12,
    'vehicle.bicycle': 13,
    'vehicle.bus.bendy': 14,
    'vehicle.bus.rigid': 15,
    'vehicle.car': 16,
    'vehicle.construction': 17,
    'vehicle.emergency.ambulance': 18,
    'vehicle.emergency.police': 19,
    'vehicle.motorcycle': 20,
    'vehicle.trailer': 21,
    'vehicle.truck': 22,
    'vehicle.ego': None
}


def get_spatiotemp_splits(nuim_train: NuImages, nclients: int) -> dict:
    """Split the training data into 10 clients based on the location and time of the data collection."""
    if nclients != 10:
        raise ValueError('The default split is designed for 10 clients.')
    random.seed(0)
    splits = {}
    # Splitting is done by log to ensure that data from the same scene is not split across clients
    for log in nuim_train.log:
        token = log['token']
        location = log['location']
        month = int(log['date_captured'].split('-')[1])
        if location == 'boston-seaport':
            if month in [3, 5]:
                splits[token] = 'client1'
            elif month in [6, 7]:
                splits[token] = 'client2'
            elif month in [8, 9]:
                splits[token] = 'client3'
            else:
                raise ValueError(f'Unexpected temporal value: {month}')
        elif location == 'singapore-onenorth' or location == 'singapore-queenstown':
            if month in [1, 2]:
                splits[token] = 'client4'
            elif month in [6, 7, 8]:
                splits[token] = random.sample(['client5', 'client6', 'client7', 'client8', 'client9'], k=1)[0]
            elif month in [9]:
                splits[token] = 'client10'
            else:
                raise ValueError(f'Unexpected temporal value: {month}')
        else:
            raise ValueError(f'Unexpected spatial value: {location}')
    return splits


def process_subset(subset: str, dataset_path: str, target_path: str, class_map: dict, nclients: int) -> None:
    """Convert annotations to YOLO format and split the data among clients for the training subset."""
    print(f'Processing subset: {subset}...')
    # Load relational database
    nuim = NuImages(dataroot=dataset_path, version=f'v1.0-{subset}', lazy=True)
    # Load data split mapping if training subset
    splits = get_spatiotemp_splits(nuim, nclients) if subset == 'train' else None
    objects_distribution = pd.read_csv(f'{target_path}/objects_distribution.csv', index_col=0)
    # Iterate over all samples (i.e. images)
    for sample_idx in tqdm(range(0, len(nuim.sample))):
        sample = nuim.get('sample', nuim.sample[sample_idx]['token'])
        log = nuim.get('log', sample['log_token'])
        destination = 'server' if splits is None else splits[log['token']]
        objects_distribution.loc['Samples', destination] += 1
        save_path = os.path.join(target_path, destination)
        img_targetpath = os.path.join(save_path, 'images', f'{sample_idx}.jpg')
        label_path = os.path.join(save_path, 'labels', f'{sample_idx}.txt')
        # Copy image to subdirectory
        img_sourcepath = dataset_path + nuim.get('sample_data', sample['key_camera_token'])['filename']
        shutil.copy(img_sourcepath, img_targetpath)
        object_tokens, surface_tokens = nuim.list_anns(sample['token'], verbose=False)
        with open(label_path, 'w') as file:
            # Iterate over all objects in the image
            for object_token in object_tokens:
                token_data = nuim.get('object_ann', object_token)
                token_name = nuim.get('category', token_data['category_token'])['name']
                if class_map[token_name] is None:
                    continue
                elif isinstance(class_map[token_name], int):
                    class_id = class_map[token_name]
                else:
                    class_id, token_name = class_map[token_name]
                # Convert to YOLO format [class_id, x, y, w, h] and write to file
                x, y, w, h = convert_bbox(
                    bbox_left=float(token_data['bbox'][0]),
                    bbox_top=float(token_data['bbox'][1]),
                    bbox_right=float(token_data['bbox'][2]),
                    bbox_bottom=float(token_data['bbox'][3]),
                    img_width=1600,
                    img_height=900
                )
                file.write(f'{class_id} {x} {y} {w} {h}\n')
                objects_distribution.loc[token_name, destination] += 1
    objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')


def process_nuimages(dataset_path: str, target_path: str, data: str, class_map: dict, nclients: int, tar: bool) -> None:
    """Convert annotations to YOLO format and splits data among the server and clients for training and validation."""
    print('Converting annotations and splitting data...')
    if class_map == 10:
        class_map = CLASS_MAP10
    elif class_map == 23:
        class_map = CLASS_MAP23
    else:
        raise ValueError(f'Unexpected class map: {class_map}')
    create_directories(target_path, nclients)
    objects_distribution = get_distribution_dataframe(data, nclients)
    objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')
    # Process training and validation subsets separately according to nuImages predefined splits
    for subset in ['train', 'val']:
        process_subset(subset, dataset_path, target_path, class_map, nclients)
    # Archive the directories of the federated participants
    if tar:
        print('Archiving...')
        archive_directories(target_path, nclients)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset-path', type=str, default='./datasets/', help='path to dataset directory')
    args.add_argument('--target-path', type=str, default=None, help='path to target directory to store processed data')
    args.add_argument('--class-map', type=int, default=10, help='map between annotations, should match yaml file')
    args.add_argument('--data', type=str, default=None, help='path to data yaml file')
    args.add_argument('--nclients', type=int, default=10, help='number of clients in federated experiment')
    args.add_argument('--tar', action='store_true', help='archive the directories of the federated participants')
    args = args.parse_args()
    data = f'./data/nuimages{args.class_map}.yaml' if args.data is None else args.data
    target_path = f'{args.dataset_path}nuimages{args.class_map}' if args.target_path is None else args.target_path
    process_nuimages(args.dataset_path, target_path, data, args.class_map, args.nclients, args.tar)
