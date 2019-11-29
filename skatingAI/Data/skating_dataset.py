import cv2
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random

path = os.getcwd()


def get_dataset():
    ds_names = get_data_names()
    ds = []

    for ds_name in ds_names:
        y_data = get_keypoints(ds_name)
        x_data = get_frames(ds_name)
        ds.append([x_data, y_data])

    return ds


def get_dataset_flat():
    ds_names = get_data_names()
    ds_x = []
    ds_y = []

    for ds_name in ds_names:
        ds_x.append(get_frames(ds_name))
        ds_y.append(get_keypoints(ds_name))

    return np.array(ds_x), np.array(ds_y)


def get_data_names():
    dirpath = Path(f"{path}/Data/KeyPoints")

    return [x.name for x in dirpath.iterdir() if x.is_dir()]


def get_keypoints(ds_name):
    ds_path = Path(f"{path}/Data/KeyPoints/{ds_name}")
    kp_json_paths = [x for x in ds_path.iterdir() if x.is_file()]
    ds_kp = []

    for file_kp in kp_json_paths:
        people = get_pose_kp(pd.read_json(file_kp.read_text()))
        ds_kp.append(people)

    return np.array(ds_kp)


def get_frames(ds_name):
    video_path = Path(f"{path}/Data/Videos").glob(f"{ds_name}.*")
    found_path = [x for x in video_path]
    frames = vid2frames(str(found_path[0]))

    return np.array(frames)


def vid2frames(video_path):
    frames = []

    video_handle = cv2.VideoCapture(video_path)

    while True:
        eof, frame = video_handle.read()
        if not eof:
            break

        frames.append(frame)

    return frames


def get_pose_kp(pose_json):
    people = []

    for person in pose_json.people:
        people.append(person['pose_keypoints_2d'])

    return np.array(people)


def get_batch(dataset, batch_size=1):
    batches = []
    for _ in range(batch_size):
        batches.append(random.choice(dataset))
    return batches
