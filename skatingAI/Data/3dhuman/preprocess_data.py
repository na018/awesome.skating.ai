from pathlib import Path
from PIL import Image
import cv2
import os
import numpy as np

path = Path.cwd()

# for x in os.walk(path):
#     print(x)

def fast_scandir(dir, process, arr=[], FileName=None):
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]

    if len(subfolders) == 0:
        path_name = os.path.relpath(dir, path).replace('/','_').replace('none_','')
        file_names = [f.path for f in os.scandir(dir) if f.is_file()]
        file_names = sorted(file_names)
        process(file_names, FileName)
        arr.append(path_name)

    for dir in list(subfolders):
        fast_scandir(dir,process, arr,FileName)

    return arr


def get_path_from_rgb(rgb_path, dir='segmentation_clothes'):
    pth = rgb_path.replace('rgb', dir)
    pth = pth.replace('jpg', 'png')
    pth = pth.replace(f"{pth.split('/')[-2]}/", '')

    return pth


def create_video(video_name, img_paths):
    frame = cv2.imread(img_paths[0])
    height, width, layers = frame.shape

    video_rgb = cv2.VideoWriter(f"{path}/processed/rgb/{video_name}.avi", 0, 30, (width, height))
    video_rgbb = cv2.VideoWriter(f"{path}/processed/rgbb/{video_name}.avi", 0, 30, (width, height))
    video_mask = cv2.VideoWriter(f"{path}/processed/mask/{video_name}.avi", 0, 30, (width, height))

    for img_pth in img_paths:
        img_rgb = cv2.imread(img_pth)
        video_rgb.write(img_rgb)

        img_clothes_pth = get_path_from_rgb(img_pth)
        img_clothes = cv2.imread(img_clothes_pth)
        img_rgb[(img_clothes == [153, 153, 153]).all(axis=2)] = [0, 0, 0]
        video_rgbb.write(img_rgb)

        img_mask_pth = get_path_from_rgb(img_pth, 'segmentation_body')
        img_mask = create_mask(img_mask_pth)
        video_mask.write(img_mask)

    cv2.destroyAllWindows()
    video_rgb.release()
    video_rgbb.release()
    video_mask.release()
class FileName():
    def __init__(self):
        self.counter = 0
    def get_name(self):
        self.counter += 1
        return self.counter

def save_np_video(img_paths, FileName):
    img_masks = []
    img_rgbs = []
    img_rgbbs = []

    counter = FileName.get_name()
    print(counter)
    if counter > 202:

        for img_pth in img_paths:
            # img_rgb = cv2.imread(img_pth)
            # img_rgbs.append(img_rgb)
            #
            # img_clothes_pth = get_path_from_rgb(img_pth)
            # img_clothes = cv2.imread(img_clothes_pth)
            # img_rgb[(img_clothes == [153, 153, 153]).all(axis=2)] = [0, 0, 0]
            # img_rgbbs.append(img_rgb)

            img_mask_pth = get_path_from_rgb(img_pth, 'segmentation_body')
            img_mask = create_mask(img_mask_pth)
            img_masks.append(img_mask)


        # np.savez_compressed(f"{path}/processed/numpy/rgbs/{counter}",img_rgbs)
        # np.savez_compressed(f"{path}/processed/numpy/rgbbs/{counter}",img_rgbbs)
        np.savez_compressed(f"{path}/processed/numpy/masks/{counter}",img_masks)


def get_segmentation_body():
  return {
    'bg': [153,153,153],
    'Head': [128,64,0],
    'RUpArm': [128, 0, 128],
    'RForeArm': [128, 128, 255],
    'RHand': [255, 128, 128],
    'LUpArm': [0, 0, 255],
    'LForeArm': [128, 128, 0],
    'LHand': [0, 128, 0],
    'torso': [128, 0, 0],
    'RThigh': [128, 255, 128],
    'RLowLeg': [255, 255, 128],
    'RFoot': [255, 0, 255],
    'LThigh': [0, 128, 128],
    'LLowLeg':  [0, 0, 128],
    'LFoot':[255, 128, 0]
    }

def create_mask(img_pth):
    img = np.asarray(Image.open(img_pth).convert('RGB'))
    body_mask = np.zeros(img.shape[:2])
    sb = get_segmentation_body()
    sb.pop('bg', None)

    for i, key in enumerate(sb.values()):
        body_mask[(img == key).all(axis=2)] = i+1

    return body_mask.astype(np.uint8)


fn = FileName()
fast_scandir(f"{path}/rgb/", save_np_video,[], fn)
