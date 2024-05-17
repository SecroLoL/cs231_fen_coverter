import typing
import cv2
import json
import numpy as np
import logging
from detect_board import find_corners, sort_corner_points
import glob
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def evaluate():
    mistakes = 0
    total = 0
    running_loss = 0
    for img_file in glob.glob("./val/*.png"):
        total += 1
        img = cv2.imread(str(img_file))
        idx = img_file.find(".png")
        stem = img_file[6:idx]
        json_file = f"./val/{stem}.json"
        with open(json_file, "r") as f:
            label = json.load(f)
        actual = np.array(label["corners"])

        try:
            predicted, _ = find_corners(img)
        except Exception:
            predicted = None

        if predicted is not None:
            actual = sort_corner_points(actual)
            predicted = sort_corner_points(predicted)
            running_loss += np.linalg.norm(actual - predicted, axis=-1)
            
        if predicted is None or np.linalg.norm(actual - predicted, axis=-1).max() > 10.:
            mistakes += 1
            
    avg_loss = np.mean(running_loss / total)
            
    return mistakes, total, avg_loss


if __name__ == "__main__":
    mistakes, total, avg_loss = evaluate()
    with open("results.txt", "w") as f:
        f.write(f"Total accuracy: {total - mistakes} correct, {mistakes} incorrect, {total} total. {(total - mistakes) / total * 100}% accuracy. Avg euclidean error in pixels: {avg_loss}")
    print(f"Total accuracy: {total - mistakes} correct, {mistakes} incorrect, {total} total. {(total - mistakes) / total * 100}% accuracy. Avg euclidean error in pixels: {avg_loss}")
    

    