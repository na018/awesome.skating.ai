import argparse
from pathlib import Path
import cv2
from matplotlib import pyplot as plt


def parse_video(video_path, img_output_dir):
    video_handle = cv2.VideoCapture(video_path)
    print(
        "width:", video_handle.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height:", video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "amount of frames:", video_handle.get(cv2.CAP_PROP_FRAME_COUNT),
        "fps:", video_handle.get(cv2.CAP_PROP_FPS),
    )
    all_frames = video_handle.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    while i < all_frames:
        eof, frame = video_handle.read()

        print(f"{img_output_dir}/frame_{i}.jpg")
        try:
            cv2.imwrite(f"{img_output_dir}/frame_{i}.jpg", frame)
        except:
            print(f"{img_output_dir}/frame_{i}.jpg is empty")

        if not eof:
            print(eof)

        i += 1


def parse_videos(video_paths, img_output_dir):
    vh = []
    for i, video_path in enumerate(video_paths):
        vh.append(cv2.VideoCapture(video_path))

    all_frames = vh[0].get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    while i < all_frames:

        fig = plt.figure(
            num=f"frame-{i}", figsize=(15, 10))

        for j in range(len(video_paths)):
            _, frame = vh[j].read()
            a = fig.add_subplot(len(video_paths), 1, j+1)
            name = video_paths[j].split('/')[-1]
            a.set_title(f"{name}")
            plt.imshow(frame,  cmap='coolwarm', interpolation='bicubic')

        plt.savefig(f"{img_output_dir}/frame_{i}.jpg")
        i += 1
        print(f"{img_output_dir}/frame_{i}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default="video/GOPR1222.MP4")
    parser.add_argument('--video_paths', nargs='+',
                        help="here you might pass multiple videos which will be combined as one frame in the output")
    parser.add_argument('--img_output_dir', default="video/GOPR1222_frames")

    args = parser.parse_args()

    if args.video_paths:
        parse_videos(args.video_paths, args.img_output_dir)
    else:
        parse_video(args.video_path, args.img_output_dir)
