import copy
import cProfile
import csv
import os
import pstats

import numpy as np
import torch
from reconchess import *

from models import Siamese_Network, get_distance
from utils import *


#constants for the input encoding
pos_opp_capture = 0
pos_last_moves = 1
pos_last_move_captured = 74
pos_last_move_None = 75
pos_own_pieces = 76
pos_sensed = 82
pos_sense_result = 83
piece_map = {'p': (0,0),'r': (0,1),'n': (0,2),'b': (0,3),'q': (0,4),'k': (0,5),'P': (1,0),
    'R': (1,1),'N': (1,2),'B': (1,3),'Q': (1,4),'K': (1,5)}

class SiameseBackend():

    def __init__(self, device = None, player_embedding = False, load_network = True, temperature = 10):
        self.network = Siamese_Network(embedding_dimensions = 512)
        self.player_embedding = player_embedding
        if not device:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        if load_network:
            path = 'Siamese_Network.pt'
            try:
                self.network.load_state_dict(torch.load(path, map_location = self.device))
            except Exception as e:
                print(e)
                raise e
            self.network = self.network.to(self.device)
            self.network.eval()
        self.board_list = []
        self.current_board = torch.zeros(90,8,8)
        self.last_sense = None
        self.own_pieces = None
        self.temperature = temperature
        self.color = None


    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        print('Siamese handled game start')
        self.color = color
        self.opponent_name = opponent_name
        if self.opponent_name in self.network.player_encoding.keys():
            print(f'Opponent {self.opponent_name} is in encodings')
        else:
            print(f'Opponent {self.opponent_name} is NOT in encodings')
        if color:
            self.current_board[-1,:,:] = 1
        self.own_pieces = copy.deepcopy(board)
        self.fill_own_pieces()

    def fill_own_pieces(self):
        for square,piece in self.own_pieces.piece_map().items():
            if piece.color == self.color:
                row,col = int_to_row_column(square)
                self.current_board[pos_own_pieces+piece_map[str(piece)][1],row,col] = 1
        

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        print('Siamese handled opponent move')
        if captured_my_piece:
            row,col = int_to_row_column(capture_square)
            self.current_board[pos_opp_capture,row,col] += 1
            self.own_pieces.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) \
            -> Optional[Square]:
        raise NotImplementedError

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        print('Siamese handled sense result')
        if self.last_sense is None:
            return
        
        if sense_result is not None:
            self.current_board[pos_sensed] += sense_location_to_area(self.last_sense)
            
            #fill in sensed pieces
            for res in sense_result:
                if res[1] is not None:
                    if str(res[1]) in piece_map:
                        c,pos = piece_map[str(res[1])]
                    else:
                        c,pos = piece_map[str(res[1]['value'])]
                    if c != self.color:
                        row,column = int_to_row_column(res[0])
                        self.current_board[pos_sense_result+pos,row,column] += 1

    def get_choice_embedding(self,choice):
        board = board_from_fen(choice.fen())
        board = board.unsqueeze(0)
        return self.network.choice_forward(board.to(self.device))

    def get_player_embedding(self,player):
        if player in self.network.player_encoding.keys():
            embedded_player = self.network.player_encoding[player].view(1,-1).to(self.device)
        else:
            embedded_player = self.network.empty_encoding.view(1,-1).to(self.device)
        return embedded_player

    def get_board_weightings(self, possible_boards):
        self.fill_own_pieces()
        num_options = len(possible_boards)
        if num_options == 0:
            print('No board options!')
            print(self.own_pieces.fen())
            return []

        #check with random board if own pieces align
        random_board = np.random.choice(list(possible_boards))
        for square,piece in random_board.piece_map().items():
            if piece.color == self.color:
                if self.own_pieces.piece_at(square) != piece:
                    print('Inconsistency between our board and the random board!')
                    print(f'Random board: {random_board.fen()}')
                    print(f'Own board: {self.own_pieces.fen()}')
                    raise Exception

        board_tensor = []
        board_list = list(possible_boards)
        for b in board_list:
            board_tensor.append(board_from_fen(b.fen()))

        board_tensor = torch.stack(board_tensor)
        board_tensors_embedded = self.network.choice_forward(board_tensor.to(self.device))

        history = self.get_history()
        anchor = self.network.anchor_forward(history.to(self.device),self.get_player_embedding(self.opponent_name),None).squeeze()
        anchor_repeated = anchor.repeat(num_options,1)
        
        distances = get_distance(anchor_repeated,board_tensors_embedded)

        weights = self.distances_to_weights(distances)
        if (torch.isnan(torch.Tensor(weights)) == True).any():
            print(distances)
            print(anchor)
            print(board_list)
            print(board_tensors_embedded)
            print(weights)
            raise Exception
        return weights,distances,anchor

    def distances_to_weights(self,distances):
        if torch.min(distances) == torch.max(distances) or distances.size(0) == 1:
            weights = torch.ones_like(distances)
            weights = weights/weights.size(0)
            return weights.tolist()
        distances += 1e-7
        softmin = torch.nn.Softmin(dim = 0)
        distances = distances/self.temperature
        distances = softmin(distances)
        distances = distances.tolist()
        return distances


    def get_embeddings(self, possible_boards):
        self.fill_own_pieces()
        num_options = len(possible_boards)
        if num_options == 0:
            print('No board options!')
            print(self.own_pieces.fen())
            return []

        #check for random board if own pieces align
        random_board = np.random.choice(possible_boards)
        for square,piece in random_board.piece_map().items():
            if piece.color == self.color:
                if self.own_pieces.piece_at(square) != piece:
                    print('Inconsistency between our board and the random board!')
                    print(random_board.fen())
                    print(self.own_pieces.fen())

        board_tensor = []
        for b in possible_boards:
            board_tensor.append(board_from_fen(b.fen()))
            


        board_tensor = torch.stack(board_tensor)
        board_tensors_embedded = self.network.choice_forward(board_tensor.to(self.device))

        history = self.get_history()
        anchor = self.network.anchor_forward(history.to(self.device),self.get_player_embedding(self.opponent_name),None).squeeze()
        return anchor_embedded,board_tensors_embedded, possible_boards
    

    def get_history(self):
        history = torch.zeros(20,90,8,8)
        history[-1,:,:,:] = self.current_board
        for i in range(19):
            if len(self.board_list) > i:
                history[-2-i] = self.board_list[-1-i]
            else:
                break
        return history.view(1,20*90,8,8)


    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        
        promoted_to_queen = False
        promoted_to_rook = False
        promoted_to_knight = False
        promoted_to_bishop = False
        self.board_list.append(self.current_board.clone())
        self.current_board = torch.zeros(90,8,8)
        if self.color:
            self.current_board[-1,:,:] = 1

        if requested_move is not None:
            requested_move = str(requested_move)
            if requested_move[-1] == 'q':
                promoted_to_queen = True
                requested_move = requested_move[:-1]
            elif requested_move[-1] == 'r':
                promoted_to_rook = True
                requested_move = requested_move[:-1]
            elif requested_move[-1] == 'n':
                promoted_to_knight = True
                requested_move = requested_move[:-1]
            elif requested_move[-1] == 'b':
                promoted_to_bishop = True
                requested_move = requested_move[:-1]
            loc = move_to_location(requested_move,self.own_pieces)
            self.current_board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        if captured_opponent_piece:
            row,col = int_to_row_column(capture_square)
            self.current_board[pos_last_move_captured,row,col] += 1
        
        
        if taken_move is None:
            self.current_board[pos_last_move_None,:,:] = 1
        else:
            if self.own_pieces.turn != self.color:
                self.own_pieces.push(chess.Move.null())

            if self.own_pieces.is_castling(taken_move):
                self.own_pieces.push(taken_move)
            else:
                piece = self.own_pieces.piece_at(taken_move.from_square)
                self.own_pieces.remove_piece_at(taken_move.from_square)
                if promoted_to_queen:
                    self.own_pieces.set_piece_at(taken_move.to_square,chess.Piece(chess.QUEEN,self.color))
                elif promoted_to_rook:
                    self.own_pieces.set_piece_at(taken_move.to_square,chess.Piece(chess.ROOK,self.color))
                elif promoted_to_knight:
                    self.own_pieces.set_piece_at(taken_move.to_square,chess.Piece(chess.KNIGHT,self.color))
                elif promoted_to_bishop:
                    self.own_pieces.set_piece_at(taken_move.to_square,chess.Piece(chess.BISHOP,self.color))
                else:
                    self.own_pieces.set_piece_at(taken_move.to_square,piece)
            







