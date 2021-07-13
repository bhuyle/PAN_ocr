import cv2
import os
import numpy as np
from tqdm import tqdm

result_txt_root_path = "./outputs/pan_vst/resnet50_VinText"
images_root_path = "./data/VietSceneText_test/images"
visualization_root_path = "./visualizations/pan_vst/resnet50_VinText"

filenames = [fn.split(".")[0] for fn in os.listdir(result_txt_root_path)]

for filename in tqdm(filenames):
    result_txt_path = os.path.join(result_txt_root_path, filename + ".txt")
    image_path = os.path.join(images_root_path, filename + ".jpg")
    visualization_path = os.path.join(visualization_root_path, filename + ".jpg")
    image = cv2.imread(image_path)

    with open(result_txt_path, "r") as fi:
        poly_points = [[int(point) for point in poly.split(",")] for poly in fi.readlines()]

        for poly in poly_points:
            x_coors = poly[::2]
            y_coors = poly[1::2]
            pair_points = np.array([[x_coors[i], y_coors[i]] for i in range(len(x_coors))], dtype=np.int32)
            cv2.drawContours(image, [pair_points], 0, (0,255,0), 2)

    cv2.imwrite(visualization_path, image)
            
