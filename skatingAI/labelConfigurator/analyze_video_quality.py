import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from skatingAI.Data.image_operations import kp2img
from skatingAI.Data.skating_dataset import get_frames, get_keypoints

video_names = ['bielmann', 'flieger_sample']


def plot_hist(video_name, scores):
    a = scores

    plt.hist(a, color='blue', edgecolor='black', bins=5)
    plt.title(f'Histogram of keypoints estimation scores for {video_name}')
    plt.xlabel('scores')
    plt.ylabel('Evaluation amount')
    plt.show()


def print_percentiles(scores):
    print("First quartile (lower 25):", np.percentile(scores, 25))
    print("Median (50):", np.percentile(scores, 50))
    print("Third quartile (75):", np.percentile(scores, 75))
    print("90th percentile:", np.percentile(scores, 90))


def sort_frames_percentile(frames, scores, keypoints):
    p_25 = np.percentile(scores, 25)
    p_50 = np.percentile(scores, 50)
    p_75 = np.percentile(scores, 75)
    frames_0, frames_25, frames_50, frames_75 = [], [], [], []

    for i, frame in enumerate(frames):

        if scores[i] >= p_75:
            frames_75.append([frame, keypoints[i][:, :2]])
        elif scores[i] >= p_50:
            frames_50.append([frame, keypoints[i][:, :2]])
        elif scores[i] >= p_25:
            frames_25.append([frame, keypoints[i][:, :2]])
        else:
            frames_0.append([frame, keypoints[i][:, :2]])

    return frames_0, frames_25, frames_50, frames_75


for _, video_name in enumerate(video_names):

    keypoints, empty_frames = get_keypoints(
        video_name, check_empty_frames=True, show_score=True)
    frames = get_frames(video_name)
    scores = np.sum(keypoints[:, :, 2], axis=1)
    print(len(frames), len(keypoints))
    print_percentiles(scores)
    plot_hist(video_name, scores)
    frames_0, frames_25, frames_50, frames_75 = sort_frames_percentile(
        frames, scores, keypoints)

    for i, [frame, kp] in enumerate(frames_75):
        img_kp = kp2img(kp, np.array(frame))
        fig = plt.figure(num=f"{video_name}_kp_analysis", figsize=(15, 10))
        plt.imshow(img_kp, interpolation='nearest')
        plt.show()
