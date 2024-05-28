import sys
import os
import torch

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
board_path = os.path.join(project_dir, 'board_detection')
occ_path = os.path.join(project_dir, 'occupancy_detection')
class_path = os.path.join(project_dir, 'piece_classifier')
fen_generation_path = os.path.join(project_dir, 'fen_generation')
sys.path.append(board_path)
sys.path.append(fen_generation_path)
sys.path.append(occ_path)
sys.path.append(class_path)

from generator import Generator
import cv2
from baseline_cnn import CNN_100 as occupancy_model
from cnn import CNN_100 as classifier_model

occ_model_path = os.path.join(occ_path, "occupancy_cnn.pt")
occ = occupancy_model()
occ.load_state_dict(torch.load(occ_model_path))

classifier_model_path = os.path.join(class_path, "pieces_cnn.pt")
cl = classifier_model()
cl.load_state_dict(torch.load(classifier_model))

g = Generator("../occupancy_detection/occupancy_cnn.pt", "../piece_classifier/pieces_cnn.pt")
img = cv2.imread("0024.png")
res = g.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(res)