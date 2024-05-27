import cv2
import numpy as np
import chess

from board_detection.detect_board import sort_corner_points

SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE

def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color, mode="NONE") -> np.ndarray:
    """Crop a chess square from the warped input image for occupancy classification.

    Args:
        img (np.ndarray): the warped input image
        square (chess.Square): the square to crop
        turn (chess.Color): the current player

    Returns:
        np.ndarray: the cropped square
    """
    if mode == "OCCUPANCY":
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        if turn == chess.WHITE:
            row, col = 7 - rank, file
        else:
            row, col = rank, 7 - file
        return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
                int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]
    elif mode == "PIECE":
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        if turn == chess.WHITE:
            row, col = 7 - rank, file
        else:
            row, col = rank, 7 - file
        height_increase = MIN_HEIGHT_INCREASE + \
            (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7)
        left_increase = 0 if col >= 4 else MIN_WIDTH_INCREASE + \
            (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3)
        right_increase = 0 if col < 4 else MIN_WIDTH_INCREASE + \
            (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3)
        x1 = int(MARGIN + SQUARE_SIZE * (col - left_increase))
        x2 = int(MARGIN + SQUARE_SIZE * (col + 1 + right_increase))
        y1 = int(MARGIN + SQUARE_SIZE * (row - height_increase))
        y2 = int(MARGIN + SQUARE_SIZE * (row + 1))
        width = x2-x1
        height = y2-y1
        cropped_piece = img[y1:y2, x1:x2]
        if col < 4:
            cropped_piece = cv2.flip(cropped_piece, 1)
        result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped_piece.dtype)
        result[OUT_HEIGHT - height:, :width] = cropped_piece
        return result
    else:
        return np.zeros((1))


def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image of the chessboard onto a regular grid.

    Args:
        img (np.ndarray): the image of the chessboard
        corners (np.ndarray): pixel locations of the four corner points

    Returns:
        np.ndarray: the warped image
    """

    src_points = sort_corner_points(corners)
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                           [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                           [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right
                           [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                           ], dtype=np.float32)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))