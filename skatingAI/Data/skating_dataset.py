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


def check_empty_frames():
    ds_names = get_data_names()

    for ds_name in ds_names:
        _, empty_frames = get_keypoints(ds_name, check_empty_frames=True)
        frames = get_frames(ds_name)
        j = 0

        print('\n'*5, '-'*100)
        if len(empty_frames > 0):
            size = frames[0].shape[:2]
            out_ef = cv2.VideoWriter(
                f"{path}/Data/EmptyFrames/{ds_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            out_rf = cv2.VideoWriter(
                f"{path}/Data/ResultingFrames/{ds_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

            for idx, frame in enumerate(frames):

                if idx > empty_frames[-1][1]:
                    continue

                print(
                    f"{[idx]}-{ds_name}: Search for empty_frame at index {empty_frames[j][1]}")

                if idx == empty_frames[j][1]:
                    cv2.imwrite(
                        f"{path}/Data/EmptyFrames/{empty_frames[j][0]}_{idx}.jpg", frame)
                    out_ef.write(frame)
                    j += 1
                else:
                    cv2.imwrite(
                        f"{path}/Data/ResultingFrames/{empty_frames[j][0]}_{idx}.jpg", frame)
                    out_rf.write(frame)

                out_ef.release()
                out_rf.release()
        print(
            f"Successfully parsed {ds_name}\nFound {len(empty_frames)} empty frames.")


def get_data_names():
    dirpath = Path(f"{path}/Data/KeyPoints")

    return [x.name for x in dirpath.iterdir() if x.is_dir()]


def get_keypoints(ds_name, check_empty_frames=False):
    ds_path = Path(f"{path}/Data/KeyPoints/{ds_name}")
    kp_json_paths = [x for x in ds_path.iterdir() if x.is_file()]
    ds_kp = []
    empty_frames = []

    for idx, file_kp in enumerate(kp_json_paths):
        people = get_pose_kp(pd.read_json(file_kp.read_text()))

        if len(people) == 0:
            empty_frames.append([ds_name, idx])

        ds_kp.append(people)

    if check_empty_frames:
        return np.array(ds_kp), empty_frames

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

    # todo: check if no people were detected -> then delete frame + kp
    # keep track how many people in subsequent frames could not be detected and analyze frames
    for person in pose_json.people:
        p = np.array(person['pose_keypoints_2d'])
        # reshape to (x,y,score)
        p = p.reshape((-1, 3))
        # delete not needed score
        p = p[:, : 2]
        people.append(p)

    return np.array(people)


def get_batch(dataset, batch_size=1):
    # todo: add functionality to split batches to equal sizes
    batches = []
    for _ in range(batch_size):
        batches.append(random.choice(dataset))
    return batches
