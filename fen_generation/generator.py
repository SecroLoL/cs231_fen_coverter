from utils import crop_square, warp_chessboard_image

class Generator:
    def __init__(self, occupancy_path, classifier_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.occupancy_model = torch.load(occupancy_path).to(device)
        self.classifier_model = torch.load(classifier_path).to(device)
        
    def classify_occupancy(self, img, turn, corners):
        warped = warp_chessboard_image(img, corners)
        squares = list(chess.SQUARES)
        square_imgs = map(functools.partial(
            crop_square, warped, turn=turn, mode="OCCUPANNCY"), squares)
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self._occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = self._occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1) == self._occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy
    
    def _classify_pieces(self, img, turn, corners, occupancy):
        occupied_squares = np.array(self._squares)[occupancy]
        warped = create_piece_dataset.warp_chessboard_image(
            img, corners)
        piece_imgs = map(functools.partial(
            crop_square, warped, turn=turn, mode="PIECE"), occupied_squares)
        piece_imgs = map(Image.fromarray, piece_imgs)
        piece_imgs = map(self._pieces_transforms, piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs)
        piece_imgs = device(piece_imgs)
        pieces = self._pieces_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        pieces = self._piece_classes[pieces]
        all_pieces = np.full(len(self._squares), None, dtype=object)
        all_pieces[occupancy] = pieces
        return all_pieces
    
    def predict(self, img, turn=chess.WHITE):
        with torch.no_grad():
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            occupancy = self._classify_occupancy(img, turn, corners)
            pieces = self._classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            return board.fen()
        
    