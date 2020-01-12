import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import random
from skimage.draw import circle
from skatingAI.Data.BODY_25_model import BODY_25
from skatingAI.Data.skating_dataset import get_data_names, get_pose_kp

path = f"{Path.cwd()}/Data"


def normalize_kp(kp, height, width):

    kp_0 = kp[0]  # x (width)
    kp_1 = kp[1]  # y(height)

    if kp_1 >= height:
        kp_1 = height - 1
    if kp_0 >= width:
        kp_0 = width-1

    return int(kp_0), int(kp_1)


def show_keypoints_img(name, frame_n, max_frame_amount=2, score_bound=18):

    imgs, img_kps,  kps, scores, frame_ns = \
        find_subsequent_frames(
            name, frame_n, max_frame_amount, score_bound)

    if imgs is None:
        return None, None, None

    max_kps, max_scores = find_max_scores(scores, kps)
    print(f"> max score: ", np.sum(max_scores))

    # plot the frames with keypoints
    fig = plt.figure(num=f"{name}_{frame_ns}_kp_analysis", figsize=(15, 10))

    for idx in range(max_frame_amount):
        # first row: frames with keypoints
        a = fig.add_subplot(2, max_frame_amount, idx+1)
        a.set_title(f"{name}_{frame_ns[idx]}_kp")
        plt.imshow(
            img_kps[idx],  cmap='coolwarm', interpolation='bicubic')

        # second row: frames with max keypoints
        a = fig.add_subplot(2, max_frame_amount, idx+max_frame_amount+1)
        a.set_title(f"{name}_{frame_ns[idx]}_result")
        img_max = kp2img(max_kps, imgs[idx])
        plt.imshow(img_max,  cmap='coolwarm', interpolation='bicubic')

    plt.show()

    return scores, max_scores, frame_ns


def find_max_scores(scores, kps):

    max_scores_idx = np.argmax(scores, axis=0)
    max_kp, max_kps = [], []

    for i, ms in enumerate(max_scores_idx):
        max_kp = kps[ms][i]
        max_kps.append([*max_kp, scores[ms][i]])

    return np.array(max_kps), np.amax(scores, axis=0)


def find_subsequent_frames(name, frame_n, max_frame_amount=2, score_bound=18):

    imgs, img_kps, scores, kps, ns = [], [], [], [], []
    subsequent_frame_n = 1
    max_frame_check_gate, max_frame_check = 100, 0

    while subsequent_frame_n <= max_frame_amount and \
            max_frame_check < max_frame_check_gate:

        img = None
        try:
            img = cv2.imread(
                f"{path}/ResultingFrames/{name}_{frame_n}.jpg", 0)
            imgs.append(img)
        except:
            print(f'{name}_{frame_n}: not found')
            subsequent_frame_n = 0
            imgs, img_kps, scores, kps, ns = [], [], [], [], []

        if img is not None:
            _img_kps, _kps, _scores = add_keypoints_img(
                name, frame_n, img.copy())
            if np.sum(_scores) > score_bound:
                img_kps.append(_img_kps)
                kps.append(_kps)
                scores.append(_scores)
                ns.append(frame_n)
            else:
                subsequent_frame_n = 0
                imgs, img_kps, scores, kps, ns = [], [], [], [], []
        else:
            subsequent_frame_n = 0
            imgs, img_kps, scores, kps, ns = [], [], [], [], []

        frame_n += 1
        subsequent_frame_n += 1
        max_frame_check += 1

    if max_frame_check == max_frame_check_gate:
        return None, None, None, None, None

    print('*'*100)
    return imgs, img_kps, kps, scores, ns


def kp2img(kp, img):

    if len(img.shape) == 2:
        height, width = img.shape
        color = 255
    else:
        height, width, rgb = img.shape
        color = np.array([255, 255, 255])

    for idx, kp in enumerate(kp):
        radius = 5
        kp_0, kp_1 = normalize_kp(kp, height-radius, width-radius)

        if len(kp) == 3:
            if kp[2] < 0.9:
                color -= 50
            if kp[2] < 0.5:
                color -= 100

        rr, cc = circle(kp_1, kp_0, radius)
        img[rr, cc] = color

    return img


def add_keypoints_img(name, frame_n, img):

    img_path = f"{path}/ResultingFrames/{name}_{frame_n}.jpg"
    kp_path = str(list(Path(
        f"{path}/KeyPoints/{name}").glob(f"{name}*0{frame_n}_keypoints.json"))[0])

    focus_person_kp = get_pose_kp(pd.read_json(kp_path), show_score=True)

    img = kp2img(focus_person_kp, img)

    return img, focus_person_kp[:, :2], focus_person_kp[:, 2]


def analyze_images():

    max_frame_amount, score_bound = 3, 19
    df_scores = pd.DataFrame(columns=["name", *BODY_25])
    df_max_scores = pd.DataFrame(columns=["name", *BODY_25])

    video_names = get_data_names()

    frames = [(name, random.randint(1, 100)) for name in video_names]

    for idx, frame in enumerate(frames):

        print("\n\n", "*"*100, "\n", frame[0])
        scores, max_scores, frames_ns = show_keypoints_img(
            frame[0], frame[1], max_frame_amount, score_bound)

        if scores is not None:

            for _i, _score in enumerate(scores):
                df_scores.loc[idx +
                              _i] = [f"{frame[0]}_{frames_ns[_i]}", *_score]

            df_max_scores.loc[idx] = [frame[0], *max_scores]
        else:
            print(
                f"\n{frame[0]} is a really crapy video! Best would be you'd throw it away very quickly.\n")

    print(df_scores)
    print(df_max_scores)
    df_scores.to_csv(f"{Path.cwd()}/Analysis_kp/scores.csv")
    df_max_scores.to_csv(f"{Path.cwd()}/Analysis_kp/max_scores.csv")


# analyze_images()
