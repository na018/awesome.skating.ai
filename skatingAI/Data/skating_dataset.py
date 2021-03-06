import cv2
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random


path = os.getcwd()


def get_dataset(batch_size=200):
    ds_names = get_data_names()
    ds = []

    for ds_name in ds_names:
        ds_x, ds_y = delete_empty_frames(
            get_frames(ds_name), get_keypoints(ds_name, check_empty_frames=True))

        batches = create_batches_of_equal_size(ds_x, ds_y, batch_size)
        [ds.append(batch) for batch in batches]

    return ds


def get_dataset_flat():
    ds_names = get_data_names()
    ds_x = []
    ds_y = []

    for ds_name in ds_names:
        x, y = delete_empty_frames(get_frames(
            ds_name), get_keypoints(ds_name, check_empty_frames=True))
        ds_x.append(x)
        ds_y.append(y)

    return np.array(ds_x), np.array(ds_y)


def check_empty_frames():
    ds_names = get_data_names()

    for ds_name in ds_names:
        _, empty_frames = get_keypoints(ds_name, check_empty_frames=True)
        frames = get_frames(ds_name)
        j = 0

        print('\n'*5, '-'*100)
        if len(empty_frames) > 0:
            height, width, _ = frames[0].shape
            out_ef = cv2.VideoWriter(
                f"{path}/Data/EmptyFrames/videos/{ds_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
            out_rf = cv2.VideoWriter(
                f"{path}/Data/ResultingFrames/videos/{ds_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

            for idx, frame in enumerate(frames):

                if idx > empty_frames[-1][1]:
                    continue

                print(
                    f"{[idx]}-{ds_name}: Search for empty_frame at index {empty_frames[j][1]}")

                if idx == empty_frames[j][1]:
                    cv2.imwrite(
                        f"{path}/Data/EmptyFrames/images/{empty_frames[j][0]}_{idx}.jpg", frame)
                    out_ef.write(frame)
                    j += 1
                else:
                    cv2.imwrite(
                        f"{path}/Data/ResultingFrames/images/{empty_frames[j][0]}_{idx}.jpg", frame)
                    out_rf.write(frame)

            out_ef.release()
            out_rf.release()
        print(
            f"Successfully parsed {ds_name}\nFound {len(empty_frames)} empty frames.")


def delete_empty_frames(frames=[], kp_empty=()):
    keypoints, empty_frames = kp_empty
    if len(empty_frames) > 0:
        keypoints = np.delete(keypoints, empty_frames[:, 1], 0)
        frames = np.delete(frames, empty_frames[:, 1], 0)

    return frames, keypoints


def get_data_names():
    dirpath = Path(f"{path}/Data/KeyPoints")

    return [x.name for x in dirpath.iterdir() if x.is_dir()]


def get_keypoints(ds_name, check_empty_frames=False, show_score=False):
    ds_path = Path(f"{path}/Data/KeyPoints/{ds_name}")
    kp_json_paths = [x for x in ds_path.iterdir() if x.is_file()]
    kp_json_paths = sorted(kp_json_paths)
    ds_kp = []
    empty_frames = []

    for idx, file_kp in enumerate(kp_json_paths):
        focus_person = get_pose_kp(
            pd.read_json(file_kp.read_text()), show_score)

        if focus_person.shape == (1, 3):
            empty_frames.append([ds_name, idx])

        ds_kp.append(focus_person)

    if check_empty_frames:
        return np.array(ds_kp), np.array(empty_frames)

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


def get_pose_kp(pose_json, show_score=False):
    people = []
    focus_person = np.zeros((1, 3))

    for person in pose_json.people:
        p = np.array(person['pose_keypoints_2d'])
        # reshape to (x,y,score)
        p = p.reshape((-1, 3))

        if np.sum(p[:, 2]) > np.sum(focus_person[:, 2]):
            focus_person = p

    # delete not needed score
    if not show_score:
        focus_person = focus_person[:, : 2]

    return focus_person


def create_batches_of_equal_size(x, y, batch_size):
    ds = []
    for i in range(0, len(x), batch_size):
        ds.append([x[i:i+batch_size:], y[i:i+batch_size:]])

    return ds


def get_batch(dataset, amount=1):
    return random.choices(dataset, k=amount)
