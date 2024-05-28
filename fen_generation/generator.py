from utils import crop_square, warp_chessboard_image, resize_image, sort_corner_points, pieces as piece_mapping
from baseline_cnn import CNN_100 as occupancy_cnn
from cnn import CNN_100 as classifier_cnn
from detect_board import find_corners
from PIL import Image
import chess
import torch
import torchvision.transforms as transforms
import numpy as np
import functools


class Generator:
    def __init__(self, occupancy_path, classifier_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(occupancy_path, classifier_path)
        self.occupancy_model = self._load_model(occupancy_path, occupancy_cnn)
        self.classifier_model = self._load_model(classifier_path, classifier_cnn)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def _load_model(self, model_path, model_class):
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model
        
    def classify_occupancy(self, img, turn, corners):
        warped = warp_chessboard_image(img, corners)
        squares = list(chess.SQUARES)
        square_imgs = map(functools.partial(
            crop_square, warped, turn=turn, mode="OCCUPANCY"), squares)
        square_imgs = map(lambda img: self.transform(img), square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs).to(self.device)
        occupancy = self.occupancy_model(square_imgs)
        occupancy = occupancy.argmax(axis=-1) == 1
        occupancy = occupancy.cpu().numpy()
        return occupancy
    
    def classify_pieces(self, img, turn, corners, occupancy):
        squares = list(chess.SQUARES)
        occupied_squares = np.array(squares)[occupancy]
        print(occupied_squares)
        if not occupied_squares:
            print("No pieces detected.")
            return np.zeros(64)
        warped = warp_chessboard_image(img, corners)
        piece_imgs = map(functools.partial(
            crop_square, warped, turn=turn, mode="PIECE"), squares)
        piece_imgs = map(lambda img: self.transform(img), piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs).to(self.device)
        
        pieces = self.classifier_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        
        # TODO: figure out what mapping was used for pieces
        pieces = piece_mapping[pieces]
        all_pieces = np.full(64, None, dtype=object)
        all_pieces[occupancy] = pieces
        return all_pieces
    
    def predict(self, img, turn=chess.WHITE):
        squares = list(chess.SQUARES)
        with torch.no_grad():
            img, img_scale = resize_image(img)
            corners = find_corners(img)
            occupancy = self.classify_occupancy(img, turn, corners)
            pieces = self.classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            return board.fen()
        
    