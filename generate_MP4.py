import cv2
import os
from tqdm import tqdm
dataset=['train','val','test']
for data in dataset:
    directory = f'/root/Data/LEVIR-MCI-dataset/images/{data}'
    file_names = os.listdir(directory)

    for file_name in tqdm(file_names):
        name_without_extension, _ = os.path.splitext(file_name)
        image1 = cv2.imread(
            f'/root/Data/LEVIR-MCI-dataset/images/{data}/A/{file_name}')
        image2 = cv2.imread(
            f'/root/Data/LEVIR-MCI-dataset/images/{data}/B/{file_name}')
        if image1.shape[:2] != image2.shape[:2]:
            raise ValueError("Images must have the same size.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'/root/Data/LEVIR-MCI-dataset/images/{data}/video/{name_without_extension}.mp4', fourcc, 2.0,
                            (image1.shape[1], image1.shape[0]))
        for i in range(8):
            weight = i / 7.0  
            interpolated_frame = cv2.addWeighted(
                image2, weight, image1, 1 - weight, 0)
            out.write(interpolated_frame)
        out.release()

