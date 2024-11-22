# coding: utf-8
# File: video_writer.py
# Description: Numpy helpers for image processing
# Created: 2024-11-20  
# Author: Gongzhe Li
import re
import  os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(image_folder, output_video_path, fps=20):
    # Get all image files from the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))



    images = []

    # Read all images
    for file in image_files:
        image_path = os.path.join(image_folder, file)
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading {image_path}")
                continue
            images.append(img)

    # Check if images were loaded
    if len(images) == 0:
        print("No images found in the folder!")
        return

    # Get the width and height from the first image
    height, width, _ = images[0].shape
    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec or 'MJPG', etc.
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write images to the video
    for img in images:
        video_writer.write(img)
    # Release the video writer object
    video_writer.release()
    print(f"Video saved at {output_video_path}")


if __name__ == '__main__':
    fps = 15
    # image_folder = '/mnt/data1/hdr_video/validation/raw_vis/night-04'
    # output_video_path = f'/mnt/data1/hdr_video/validation/raw-night-04-{fps}.avi'

    image_folder = '/home/gongzheli/workspace/mmdetection/vis_rgb_day02/vis'
    output_video_path = f'/mnt/data1/hdr_video/validation/isp-day-02-{fps}.avi'
    main(image_folder, output_video_path, fps=fps)

    # image_folder = '//mnt/data1/hdr_video/validation/raw_vis/day-03'
    # output_video_path = f'/mnt/data1/hdr_video/validation/raw-day-03-{fps}.avi'
    # main(image_folder, output_video_path, fps=fps)
    pass
