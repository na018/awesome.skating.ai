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
        self.path_processed_dir = f"{os.getcwd()}/skatingAI/Data/3dhuman/processed"
        self.dataset_url = "https://cv.iri.upc-csic.es/Dataset/"
        self.labels = ["rgb", "segmentation_clothes", "segmentation_body"]
        self.file_count = 0

        if chunk_amount < 1 or chunk_amount * 5 > 40:
            raise AssertionError(f"ChunkNumber must be between 1 and 8 but was {chunk_amount}")

        self._create_processed_folders()

    def _create_processed_folders(self):
        base = f"{self.path_processed_dir}/numpy/"
        folders = ["rgb", "rgbb", "masks"]

        for folder in folders:
            if not os.path.exists(base + folder):
                print(f"create new folder: {base + folder}")
                os.makedirs(base + folder)

    def _delete_dirs(self):

        subfolders = [f.path for f in os.scandir(self.path_processed_dir) if f.is_dir()]
        files = [f.path for f in os.scandir(self.path_processed_dir) if f.is_file()]

        for folder in subfolders:
            if "numpy" not in folder:
                shutil.rmtree(folder)
                print(f"successfully deleted [{folder}]")

        for file in files:
            os.remove(file)
            print(f"successfully deleted [{file}]")

    def download_data(self):
        max_chunks = self.chunk_amount * 5

        for i in range(1, max_chunks, 5):
            for label in self.labels:
                for chunk in ['woman', 'man']:
                    # download data
                    self._thread_download_data(label, chunk, i)
                    # process data
            self._scandir4img()
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
        os.system(f"tar -xzf {save_url} -C {self.path_processed_dir}")

    def _scandir4img(self, dir: str = f"{os.getcwd()}/skatingAI/Data/3dhuman/processed/rgb",
                     arr: List[List[str]] = []):
        subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]

        if len(subfolders) == 0:
            file_names = [f.path for f in os.scandir(dir) if f.is_file()]

            arr.append(self._read_imgs2numpy(file_names))

        for dir in list(subfolders):
            self._scandir4img(dir, arr)

        return arr

    def _read_imgs2numpy(self, file_names: List[str]):
        all_imgs = {'masks': [], 'rgb': [], 'rgbb': []}
        for name in sorted(file_names):
            file = name.replace(f"/{name.split('/')[-2]}", '')
            file = file.replace("jpg", 'png')
            segmentation_body_path = file.replace(f"rgb", 'segmentation_body')
            segmentation_clothes_path = file.replace(f"rgb", 'segmentation_clothes')

            rgb_img = cv2.imread(name)
            segmentation_body_img = self._preprocess_img2classes(
                np.asarray(Image.open(segmentation_body_path).convert('RGB')))
            segmentation_clothes_img = cv2.imread(segmentation_clothes_path)
            rgbb_img = self._create_rgbb(rgb_img, segmentation_clothes_img)

            all_imgs['rgb'].append(rgb_img)
            all_imgs['rgbb'].append(rgbb_img)
            all_imgs['masks'].append(segmentation_body_img)

        for img_sequence in all_imgs:
            np.savez_compressed(f"{self.path_processed_dir}/numpy/{img_sequence}/{self.file_count}",
                                all_imgs[img_sequence])

        self.file_count += 1

        print(f"{self.file_count} saved {file_names[0].split('/')[-2]} mask and rgb as compressed npz.")

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


DataAdmin(8).download_data()
