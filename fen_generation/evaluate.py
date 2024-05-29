import sys
import os
import torch
import glob
import chess
import numpy as np
from chess import Status
import json

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

def _get_num_mistakes(groundtruth: chess.Board, predicted: chess.Board) -> int:
    groundtruth_map = groundtruth.piece_map()
    predicted_map = predicted.piece_map()
    return sum(0 if groundtruth_map.get(i, None) == predicted_map.get(i, None) else 1
               for i in chess.SQUARES)


def _get_num_occupancy_mistakes(groundtruth: chess.Board, predicted: chess.Board) -> int:
    groundtruth_map = groundtruth.piece_map()
    predicted_map = predicted.piece_map()
    return sum(0 if (i in groundtruth_map) == (i in predicted_map) else 1
               for i in chess.SQUARES)


def _get_num_piece_mistakes(groundtruth: chess.Board, predicted: chess.Board) -> int:
    groundtruth_map = groundtruth.piece_map()
    predicted_map = predicted.piece_map()
    squares = filter(
        lambda i: i in groundtruth_map and i in predicted_map, chess.SQUARES)
    return sum(0 if (groundtruth_map.get(i) == predicted_map.get(i)) else 1
               for i in squares)
    
    
def evaluate(generator, output_file, dataset_folder):
    output_file.write(",".join(["file",
                                "num_incorrect_squares",
                                "occupancy_classification_mistakes",
                                "piece_classification_mistakes",
                                "actual_num_pieces",
                                "predicted_num_pieces",
                                *(["fen_actual", "fen_predicted", "fen_predicted_is_valid"])]) + "\n")
    print(dataset_folder + "/*.png")
    for i, img_file in enumerate(glob.glob(dataset_folder + "/*.png")):
        img = cv2.imread(str(img_file))
        idx = img_file.find(".png")
        stem = img_file[:idx]
        json_file = f"{stem}.json"
        with open(json_file, "r") as f:
            label = json.load(f)

        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        groundtruth_board = chess.Board(label["fen"])
        predicted_board = generator.predict(img)
        mistakes = _get_num_mistakes(groundtruth_board, predicted_board)
        occupancy_mistakes = _get_num_occupancy_mistakes(
            groundtruth_board, predicted_board)
        piece_mistakes = _get_num_piece_mistakes(
            groundtruth_board, predicted_board)
        file_id = stem[idx - 4:]
        output_file.write(",".join(map(str, [file_id,
                                             mistakes,
                                             occupancy_mistakes,
                                             piece_mistakes,
                                             len(groundtruth_board.piece_map()),
                                             len(predicted_board.piece_map()),
                                             *([groundtruth_board.board_fen(),
                                                predicted_board.board_fen(),
                                                predicted_board.status() == Status.VALID])])) + "\n")
        if (i+1) % 5 == 0:
            output_file.flush()
            print(f"Processed {i+1} files from {dataset_folder}")
            
if __name__ == "__main__":
    with open("generator_eval.txt", "w") as f:
        evaluate(g, f, "/Users/brentju/Desktop/CS231n/cs231_fen_coverter/data/test")