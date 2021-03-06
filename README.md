# TetrisAI
The following is an algorithm that learns to play Tetris.  It does so by utilizing genetic algorithms.  It calculates the state of each board by four heuristics: the total height of all the pieces in every column, the number of lines that could be potentially cleared, the number of gaps created, and the cumulative absolute height difference between adjacent columns.  

The algorithm iterates through all rotations of the piece in every possible position and calculates a score for every permutation based on the values of the aforementioned heuristics multiplied by weights.  From here, the highest score is selected.  The purpose of the genetic algorithm is to learn which weights correspond to the best decision making.

After 80 trials, it converged to the following set of weights: -0.52, 0.55, -0.64, -0.15
