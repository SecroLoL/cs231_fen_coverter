import sys
import os
import torch

# sys.path.append(board_path)
# sys.path.append(fen_generation_path)
# sys.path.append(occ_path)
# sys.path.append(class_path)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
board_path = os.path.join(project_dir, 'board_detection')
occ_path = os.path.join(project_dir, 'saved_models', 'occupancy')
class_path = os.path.join(project_dir, 'saved_models', 'pieces')
fen_generation_path = os.path.join(project_dir, 'fen_generation')

from fen_generation.generator import Generator
import cv2
from occupancy_detection.baseline_cnn import CNN_100 as occupancy_model
from piece_classifier.cnn import CNN_100 as classifier_model

occ_model_path = os.path.join(occ_path, "occupancy_cnn.pt")
occ = torch.load(occ_model_path) 

classifier_model_path = os.path.join(class_path, "pieces_cnn.pt")
cl = torch.load(classifier_model_path)

g = Generator(occ_model_path, classifier_model_path)

img_path = os.path.join(os.path.dirname(__file__), "0024.png")
img = cv2.imread(img_path)
res = g.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(res)