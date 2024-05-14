# Chess board FEN converter
Convert images of chess board positions into FEN notation for further engine analysis 

# Overview
We attempt to build a computer vision system that is capable of converting images of **physical chess boards and pieces** to Forsyth-Edwards Notation (FEN). FEN is a compact representation of a chess position used to communicate and analyze the state of a chess game. It is challenging for existing systems to analyze the state of an over-the-board chess game, motivating a FEN-conversion system which enables this analysis via chess engines. To accomplish this goal, we construct a model, internally using **three different vision models** to detect squares, determine the occupancy of each square, and finally classify which piece belongs in each square. We experiment with different open-source methods for each subtask, including OpenCV for square detection, InceptionV3 for occupancy detection, and OWLv2 for piece classification. After these three stages, we use a mapping algorithm between piece location and piece classification to systematically generate FEN for the board position. We train our model variants on a 3-D visual engine-generated dataset of nearly 5,000 chess positions.

The dataset we use to train and evaluate the model can be found here: https://osf.io/xf3ka/wiki/home/

