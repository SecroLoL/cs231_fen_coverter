# Chess board FEN converter
Convert images of chess board positions into FEN notation for further engine analysis 

# Overview
We attempt to build a computer vision system that is capable of converting images of **physical chess boards and pieces** to Forsyth-Edwards Notation (FEN). FEN is a compact representation of a chess position used to communicate and analyze the state of a chess game. It is challenging for existing systems to analyze the state of an over-the-board chess game, motivating a FEN-conversion system which enables this analysis via chess engines. To accomplish this goal, we construct a model, internally using **three different vision models** to detect squares, determine the occupancy of each square, and finally classify which piece belongs in each square. We experiment with different open-source methods for each subtask, including OpenCV for square detection, InceptionV3 for occupancy detection, and OWLv2 for piece classification. After these three stages, we use a mapping algorithm between piece location and piece classification to systematically generate FEN for the board position. We train our model variants on a 3-D visual engine-generated dataset of nearly 5,000 chess positions.

# Model stages 

## Board detection
For our board detection stage, we use a layered approach of mathematical algorithms to compute the four corners of the board and subsequently divide it evenly into the 64 squares of the chess board. 
Specifically, we separate the board's edges from any edges detected from pieces via Hough transform and then run the RANSAC algorithm to generate a transformation matrix to provide us with a bird's-eye view of the board. From there, we use OpenCV's corner-finder method to identify the boundaries of the square board. This finally enables us to compute where each square is with some simple arithmetic over the pixels.

## Occupancy classification
We formulate our occupancy classification problem as a binary classification task on images of cropped chess board squares as identified by the board detection stage.
We offer a range of models for occupancy detection on board squares. Our models include a baseline CNN composed of three convolutional layers, three pooling layers, and three fully connected layers.
Additionally, we have more advanced models that we finetune on, such as Resnet, and InceptionV3. The code to train and evaluate these models can be found in `occupancy_detection/`.

## Piece classification
Once a board square has been identified as containing a piece, we run a piece classifier over the cropped image to classify which color and type of piece the piece is. 
We formulate the piece classification stage as a multiclass classification task where we attempt to assign a (color, piece) pair to every image.
We offer a range of models for this task, such as our baseline CNN with the same model architecture as the CNN for occupancy detection, except that the classification head is changed 
to accommodate the larger number of classes. We also finetune larger model architectures that are pretrained on ImageNet, such as ResNet, InceptionV3, and the Vision Transformer (ViT) (work-in-progress).
The code for training and evaluating these models can be found in `piece_classifier/`.
