import os
import shutil
import subprocess
from typing import List

import cv2
import numpy as np
from PIL import Image

from skatingAI.utils.utils import BodyParts, segmentation_class_colors, body_part_classes


class DataAdmin(object):
    def __init__(self, chunk_amount: int = 1):
        self.chunk_amount = chunk_amount
        self.path_processed_dir = f"{os.getcwd()}/processed"
        self.dataset_url = "https://cv.iri.upc-csic.es/Dataset/"
        self.labels = ["rgb", "segmentation_clothes", "segmentation_body", "skeleton"]
        self.file_count = 0

        if chunk_amount < 1 or chunk_amount * 5 > 40:
            raise AssertionError(f"ChunkNumber must be between 1 and 8 but was {chunk_amount}")

        self._create_processed_folders()

    def _create_processed_folders(self):
        base = f"{self.path_processed_dir}/numpy/"
        folders = ["rgb", "rgbb", "masks", "skeletons"]

        for folder in folders:
            if not os.path.exists(base + folder):
                print(f"create new folder: {base + folder}")
                os.makedirs(base + folder)

    def _delete_dirs(self):
        print("\n...start to delete folders")
        subfolders = [f.path for f in os.scandir(self.path_processed_dir) if f.is_dir()]
        files = [f.path for f in os.scandir(self.path_processed_dir) if f.is_file()]

        for file in files:
            os.remove(file)
            print(f"successfully deleted [{file}]")

        for folder in subfolders:
            if "numpy" not in folder:
                shutil.rmtree(folder)
                print(f"successfully deleted [{folder}]")



    def process_data(self):
        """ just process data, if download and extraction was already successful """
        max_chunks = self.chunk_amount * 5

        for i in range(1, max_chunks, 5):
            self._scandir4img()
            #self._delete_dirs()
            print("all data was successfully downloaded")

    def download_and_process_data(self):
        max_chunks = self.chunk_amount * 5

        for i in range(1, max_chunks, 5):
            for label in self.labels:
                for chunk in ['woman', 'man']:
                    # download data
                    self._thread_download_data(label, chunk, i)
                    # process data
            success = self._scandir4img()
            if success:
                self._delete_dirs()

        print("all data was successfully downloaded")

    def _thread_download_data(self, label: str, chunk: str, i: int):
        # download chunk with wget
        file_name = f"{label}_{chunk}{i:02d}_{i + 4:02d}.tar.gz"
        download_chunk_url = f"{self.dataset_url + label}/{file_name}"
        save_url = f"{self.path_processed_dir}/{file_name}"

        if not os.path.exists(save_url):
            print(f"start to download {download_chunk_url} to {save_url}")
            print("this may take a while...")
            proc = subprocess.Popen(["wget", '-q', download_chunk_url, '-O', save_url])
            n = proc.wait()
            print(f"downloaded: {i}")

        # extract chunk
        os.system(f"tar -xzkf {save_url} -C {self.path_processed_dir}")

    def _scandir4img(self, dir: str = f"{os.getcwd()}/processed/rgb"):

        chunks_women_men = [f.path for f in os.scandir(dir) if f.is_dir()]
        if len(chunks_women_men) < 10:
            print(f"There are only {len(chunks_women_men)} folders. Something must be wrong.")
            return False
        # paths_videos = []
        for dir in list(chunks_women_men):
            paths_videos = [f.path for f in os.scandir(dir) if f.is_dir()]
            for dir_video in list(paths_videos):
                camera_dirs = [f.path for f in os.scandir(dir_video) if f.is_dir()]
                for cam in camera_dirs:
                    cam_sub_dir = [f.path for f in os.scandir(cam) if f.is_dir()][0]
                    segmentation_body_dir = cam.replace(f"rgb", 'segmentation_body')
                    segmentation_clothes_dir = cam.replace(f"rgb", 'segmentation_clothes')
                    skeleton_dir = cam.replace(f"rgb", 'skeleton')

                    if os.path.exists(segmentation_body_dir) and \
                            os.path.exists(segmentation_clothes_dir) and \
                            os.path.exists(skeleton_dir):
                        self._read_imgs2numpy(cam_sub_dir, segmentation_body_dir, segmentation_clothes_dir,
                                              skeleton_dir)
                    else:
                        print(f"{cam_sub_dir.split('/')[-3:]} does not exist for all components.")

        return True

    def _read_imgs2numpy(self, rgb_dir, segmentation_body_dir, segmentation_clothes_dir, skeleton_dir):
        # check weather directories exist for masks, segmentation_body, segmentation_clothes, skeleton
        all_imgs = {'masks': [], 'rgb': [], 'rgbb': [], 'skeletons': []}
        file_names_rgb = [f.path for f in os.scandir(rgb_dir) if f.is_file()]
        video_name = '__'.join(segmentation_body_dir.split('/')[-3:-1])
        camera_name = segmentation_body_dir.split('/')[-1]

        for file_name_rgb in sorted(file_names_rgb):
            file_name = file_name_rgb.split('.')[-2].split('/')[-1]
            file_name_segmentation_body = f"{segmentation_body_dir}/{file_name}.png"
            file_name_segmentation_clothes = f"{segmentation_clothes_dir}/{file_name}.png"
            file_name_skeleton = f"{skeleton_dir}/{file_name}.txt"

            if os.path.exists(file_name_rgb) and \
                    os.path.exists(file_name_segmentation_body) and \
                    os.path.exists(file_name_segmentation_clothes) and \
                    os.path.exists(file_name_skeleton):
                rgb_img = cv2.imread(file_name_rgb)
                segmentation_clothes_img = cv2.imread(file_name_segmentation_clothes)

                all_imgs['rgb'].append(cv2.imread(file_name_rgb))
                all_imgs['rgbb'].append(self._create_rgbb(rgb_img, segmentation_clothes_img))
                all_imgs['masks'].append(self._preprocess_img2classes(
                    np.asarray(Image.open(file_name_segmentation_body).convert('RGB'))))
                skeleton_arr = np.loadtxt(file_name_skeleton)[:, :2][
                    [0, 1, 5, 9, 10, 11, 12, 33, 34, 35, 36, 57, 58, 59, 61, 62, 63, 64, 66]]
                skeleton_arr = np.reshape(skeleton_arr, -1)
                all_imgs['skeletons'].append(skeleton_arr)

            else:
                print(f"{file_name_rgb.split('/')[-3:]} does not exist for all components.")

        for img_sequence in all_imgs:
            np.savez_compressed(f"{self.path_processed_dir}/numpy/{img_sequence}/{video_name}_{camera_name}",
                                all_imgs[img_sequence])

        self.file_count += 1

        print(f"[{self.file_count}] saved `{video_name}:{camera_name}` mask and rgb as compressed npz.")

        return all_imgs

    def _preprocess_img2classes(self, img):
        body_mask = np.zeros(img.shape[:2])
        sb = {**segmentation_class_colors}
        sb.pop(BodyParts.bg.name)

        for i, key in enumerate(sb):
            body_mask[(img == sb[key]).all(axis=2)] = body_part_classes[key]

        return body_mask.astype(np.uint8)

    def _create_rgbb(self, rgb_img, segmentation_clothes_img):
        img = rgb_img.copy()
        img[(segmentation_clothes_img == [153, 153, 153]).all(axis=2)] = [0, 0, 0]

        return img


DataAdmin(chunk_amount=8).download_and_process_data()
