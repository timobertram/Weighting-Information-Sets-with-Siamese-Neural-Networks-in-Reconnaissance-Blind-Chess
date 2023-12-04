# Neural-Network-based-Information-Set-Weighting-for-Playing-Reconnaissance-Blind-Chess
This codebase was used for the paper "Neural-Network-based-Information-Set-Weighting-for-Playing-Reconnaissance-Blind-Chess", which is in review for IEEE Transactions on Games 2023, and provides the code required for training your own Siamese neural network for Reconnaissance Blind Chess (RBC) from scratch, as well as a trained network that you can use in your own RBC agent.

This repository is seperated into two parts: the code that was used to train a Siamese neural network on RBC game data, as well as the RBC agent used for evaluation of the final network. As this is in extension of our previous paper "Weighting-Information-Sets-with-Siamese-Neural-Networks-in-Reconnaissance-Blind-Chess" that was accepted at IEEE Conference on Games 2023, a lot of the code overlaps with our repository at https://github.com/timobertram/Weighting-Information-Sets-with-Siamese-Neural-Networks-in-Reconnaissance-Blind-Chess.

# Usage for playing RBC

If you only want to use the trained agent for playing, just use SiameseOptimist.py. In order for this to work, you need to change the variable path_to_stockfish to the path of your Stockfish executable file.

# Usage for weighting information sets of board positions

If you want to use the Siamese network in your own agent, you only need SiameseBackend.py and the trained network. The SiameseBackend is able to track the observations on its own as long as you keep calling handle_game_start, handle_opponent_move_result, handle_sense_result, and handle_move_result, and only requires a list of the possible boards in get_board_weightings(possible_boards). This will return two lists, weights and distances, distances being the raw distances in the embedding space and and weights being the transformed distances (you should in most cases only use weights).


# Usage for training

If you want to retrain the Siamese network, follow these steps:

1. Download RBC game data from https://rbc.jhuapl.edu/about and put the .json files in a folder "data/games/"
2. Run create_data.py. This will run a multiprocessed script that converts the game files into training data for the network and create a folder "data/siamese/" which contains three folders; finished_files, train, and val. finished_files is used in case you need to restart create_data.py and tracks which .json files were already processed, you can delete this folder afterwards. train and val each contain two subfolders, move and sense, which split the training data into the two decision-points of each turn. This process can take a while!
3. Run training.py

