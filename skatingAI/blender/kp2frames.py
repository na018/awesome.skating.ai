from skatingAI.Data.image_operations import kp2img
from skatingAI.Data.skating_dataset import vid2frames
import numpy as np
import json
from matplotlib import pyplot as plt
import cv2
import os

kp_path = '/home/nadin-katrin/awesome.skating.ai/keypoint_data/keypointsvv4.json'
vid_path = '/home/nadin-katrin/Pictures/FigureSkatingDemo.mkv'
video_name = '/home/nadin-katrin/Pictures/FigureSkatingDemo_annotated.mkv'

frames = vid2frames(vid_path)


with open(kp_path) as json_file:
    kps = json.load(json_file)

width, height = 1920, 1080

video = cv2.VideoWriter(
    video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))


for i in range(len(kps)):
    img = frames[i]
    kp = np.array(kps[i])
    new_img = kp2img(kp, img)
    video.write(new_img)

cv2.destroyAllWindows()
video.release()
