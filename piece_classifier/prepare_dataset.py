import os
import shutil
import random
import cv2
import chess
import numpy as np
import json
from PIL import Image

def split_files_into_directories(src_dir, num_dirs=5):
    # Create destination directories if they don't exist
    dest_dirs = [os.path.join(src_dir, f"split_{i+1}") for i in range(num_dirs)]
    for dest_dir in dest_dirs:
        os.makedirs(dest_dir, exist_ok=True)
    
    # Get a list of files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # Shuffle the files to ensure randomness
    random.shuffle(files)
    
    # Distribute files into the destination directories
    for i, file in enumerate(files):
        dest_dir = dest_dirs[i % num_dirs]
        shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))
    
    print(f"Files have been split into {num_dirs} directories.")

# Usage example
src_directory = "/Users/alexshan/Desktop/chesscog/data/occupancy/train/empty"
split_files_into_directories(src_directory)


# CHESS DATASET
SQUARE_SIZE = 50
BOARD_SIZE = 400  # pixels
IMG_SIZE = 50 * 402  # pixels

# Points where the board should be
dst_points = np.array([
    [SQUARE_SIZE, SQUARE_SIZE],  # top left
    [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
    [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE],  # bottom right
    [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
], dtype=np.float32)


def take_square_surrounding(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    # Crop a chess square which will be used for occupancy classification 
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    row, col = rank, 7 - file
    if turn == chess.WHITE:
        row, col = 7 - rank, file

    row_start = int(SQUARE_SIZE * (row + 0.5))
    row_end = int(SQUARE_SIZE * (row + 1.5))
    col_start = int(SQUARE_SIZE * (col + 0.5))
    col_end = int(SQUARE_SIZE * (col + 1.5))

    return img[row_start:row_end, col_start:col_end]

def shift_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    # Convert chessboard image to be projected onto rectangular shape
    
    src_points = sort_corners(corners)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))

def get_all_squares_labels(output_dir: str = "output", img_path = "", json_path = ""):
    """
    takes squares from each sample picture in the Input DIR and extracts the labels with them cropped.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    OCCUPIED, EMPTY = "occupied", "empty"
    
    with open(json_path, "r") as f:
        label = json.load(f)

    corners = np.array(label["corners"], dtype=np.float32)
    unwarped = shift_image(img, corners)
    board = chess.Board(label["fen"])

    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        target_class = OCCUPIED if piece is not None else EMPTY

        piece_img = take_square_surrounding(unwarped, square, label["white_turn"])

        output_path = os.path.join(output_dir, i, target_class)
        Image.fromarray(piece_img, "RGB").save(output_path)

def sort_corners(corners: np.ndarray) -> np.ndarray:
    """Sort the corner points in the order of top-left, top-right, bottom-right, bottom-left."""
    sorted_corners = np.zeros((4, 2), dtype=np.float32)
    s = np.sum(corners, axis=1)
    diff = np.diff(corners, axis=1)

    sorted_corners[0] = corners[np.argmin(s)]
    sorted_corners[2] = corners[np.argmax(s)]
    sorted_corners[1] = corners[np.argmin(diff)]
    sorted_corners[3] = corners[np.argmax(diff)]

    return sorted_corners

