import chess.engine
import reconchess as rbc
import os
from utils import *
from random import shuffle
import torch
from board_dict import BoardDict
import copy
from collections import defaultdict
import time
from SiameseBackend import *


STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
#change this to your stockfish path
path_to_stockfish = 'path/to/stockfish'

class SiameseAgent(Player):
    def __init__(self):
        self.board_dict = BoardDict()
        self.color = None
        self.first_turn = True
        self.temperature = 10

        #change this to your stockfish path
        stockfish_path = "stockfish/src/stockfish"
        stockfish_path = 'C:/stockfish/stockfish-windows-x86-64-avx2.exe'
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True, timeout = 120)
        except:
            raise Exception('Could not open stockfish')


        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.siamese = SiameseBackend(device = self.device, load_network = True, temperature = self.temperature)

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board_dict.add_board(board)
        self.color = color
        self.siamese.handle_game_start(color,board,opponent_name)
        self.opponent_name = opponent_name


    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        start_time = time.time()
        len_boards_before = self.board_dict.size()
        max_board_num = 25_000

        current_boards = self.board_dict.get_boards()
        shuffle(current_boards)
        # for some reasons the game tries to notify of opponent turn on first move for white so ignore that
        if self.first_turn and self.color:
            self.first_turn = False
            return


        self.siamese.handle_opponent_move_result(captured_my_piece,capture_square)
        resulting_boards = BoardDict()
        # if the opponent did not capture a piece, account for all possible moves on all boards
        if not captured_my_piece:
            
            # passing is an option so move old boards over
            for board in current_boards:
                new_board = board.copy()
                new_board.push(chess.Move.null())
                new_board.last_e_to_square = None  # we append None since we do not have a "last move"
                new_board.clear_stack()
                resulting_boards.add_board(new_board)

            # save fens so that we don't have multiples of the same board
            fens = [reduce_fen(board.fen()) for board in resulting_boards.get_boards()]
            for board in current_boards:
                if resulting_boards.size() > max_board_num:
                    break
                # iterate through all moves:
                for move in list(board.pseudo_legal_moves):
                    # exclude taking moves
                    if board.piece_at(move.to_square) is None and not board.is_en_passant(move):
                        new_board = board.copy()
                        new_board.push(move)
                        new_board.clear_stack()
                        new_fen = reduce_fen(new_board.fen())
                        if new_fen not in fens:
                            resulting_boards.add_board(new_board)
                            new_board.last_e_to_square = move.to_square
                            fens.append(new_fen)
                # castling:
                for board in illegal_castling(board, self.color):
                    board.clear_stack()
                    new_fen = reduce_fen(board.fen())
                    if new_fen not in fens:
                        resulting_boards.add_board(board)
                        fens.append(new_fen)
        # if a piece was captured, other moves have to be accounted for
        else:
            fens = []
            for board in current_boards:
                # board.turn = not self.color
                for move in board.pseudo_legal_moves:
                    # only look at the moves which captured on the given square
                    if move.to_square == capture_square:
                        new_board = board.copy()
                        new_board.push(move)
                        new_board.clear_stack()
                        new_fen = reduce_fen(new_board.fen())
                        if new_fen not in fens:
                            resulting_boards.add_board(new_board)
                            new_board.last_e_to_square = move.to_square
                            fens.append(new_fen)
                    #if the enemy can en passant 
                    if board.is_en_passant(move):
                        if self.color:
                            if not move.to_square+8 == capture_square:
                                continue
                        else:
                            if not move.to_square - 8 == capture_square:
                                continue
                        new_board = board.copy()
                        new_board.push(move)
                        new_board.clear_stack()
                        new_fen = reduce_fen(new_board.fen())
                        if new_board.piece_at(capture_square) is None and new_fen not in fens:
                            resulting_boards.add_board(new_board)
                            new_board.last_e_to_square = move.to_square
                            fens.append(new_fen)

        self.board_dict = resulting_boards


    def board_conflicts(self, weighted_boards):
        good_senses = [9,10,11,12,13,14,17,18,19,20,21,22,25,26,27,28,29,30,33,34,35,36,37,38,41,42,43,44,45,46,49,50,51,52,53,54]
        squares = np.zeros(64)
        relevant_squares = [index for index in range(64) if not weighted_boards[0][0].piece_at(index) or
                              weighted_boards[0][0].piece_at(index).color is not self.color]
        elimination_chances = []
        for square in good_senses:
            sense_squares = get_adjacent_squares(square)
            square_weights = defaultdict(int)
            square_num = defaultdict(int)
            elimination_chances_square = defaultdict(list)
            for board,weight in weighted_boards:
                board_string = sense_result_to_string(board,sense_squares,relevant_squares)
                square_weights[board_string] += weight
                square_num[board_string] += 1
                elimination_chances_square[board_string].append(reduce_fen(board.fen()))
            total_weight = sum(square_weights.values())
            total_boards = sum(square_num.values())
            if total_weight == 0:
                total_weight = 1
            elimination_chances_square = [(boards,square_weights[res]/total_weight) for res,boards in elimination_chances_square.items()]
            elimination_chances.append(elimination_chances_square)
            res = [(v/total_weight) * (total_boards-square_num[k]) for k,v in square_weights.items()]
            squares[square] = sum(res)
        return squares,elimination_chances

    def sense_weighted_reduction(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) \
            -> Optional[Square]:

        start_time = time.time()
        weighted_boards = self.reduce_boards(100)
        sense_values,_ = self.board_conflicts(weighted_boards)

        #Output            
        sense_values = sense_values.reshape(64)
        return sense_values

    def get_turn_number(self):
        return len(self.siamese.board_list) + 1

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) \
            -> Optional[Square]:
        num_boards = self.board_dict.size()
        sense_values = self.sense_weighted_reduction(sense_actions,move_actions,seconds_left)
        resulting_sense = np.argmax(sense_values)
        self.siamese.last_sense = resulting_sense
        return sense_actions[resulting_sense]

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        len_boards_before = self.board_dict.size()
        self.siamese.handle_sense_result(sense_result)
        # removes all boards that are not fitting the sensing result
        self.board_dict.delete_boards(sense_result)

        
        # checks if our sensing was good
        print(f'Sense. Boards before handle_sense_result: {len_boards_before}, after:  {self.board_dict.size()} against {self.opponent_name}')
        if self.board_dict.size() < 1:
            print('Not possible boards, this should not happen')
     

    def get_best_move_of_board(self,board,limit = chess.engine.Limit(time=0.1)):
        enemy_king_square = board.king(not self.color)
        # if there are any ally pieces that can take king, execute one of those moves
        enemy_king_attackers = board.attackers(self.color, enemy_king_square)
        if enemy_king_attackers:
            attacker_square = enemy_king_attackers.pop()
            move = chess.Move(attacker_square, enemy_king_square)
        else:
            try:
                move = self.engine.play(board,limit).move
            except:
                print('Something bad happened when choosing a move')
                print(board)
                move = chess.Move.null()
        return move

    
    def average_evaluation_move(self,move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        ranking = self.reduce_boards(50)
        move_ratings = defaultdict(list)
        moves = [self.get_best_move_of_board(board) for board,_ in ranking]
        for move in moves:
            for board,weight in ranking:
                tmp_board = copy.deepcopy(board)
                if not tmp_board.is_legal(move):
                    tmp_board.push(chess.Move.null())
                    tmp_board.clear_stack()
                else:
                    tmp_board.push(move)
                move_ratings[move].append(self.eval_of_move(move, board)*weight)
        best_move = max(move_ratings,key = lambda k:sum(move_ratings[k]))
        print(f'Best move is {best_move} with an eval of {sum(move_ratings[best_move])}')
        return max(move_ratings,key = lambda k:sum(move_ratings[k]))

    def eval_of_move(self, move, board):
        tmp_board = copy.deepcopy(board)
        turn = tmp_board.turn
        if board.attackers(turn, board.king(not turn)):
            return 1    
        if board.attackers(not turn, board.king(turn)):
            if not board.is_legal(move):
                return 0
        if not tmp_board.is_legal(move):
            tmp_board.turn = not tmp_board.turn
        else:
            tmp_board.push(move) 
        return 1-stockfish_eval(board = tmp_board, engine = self.engine) 

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        print(f'{seconds_left} seconds left against {self.opponent_name}')
        return self.average_evaluation_move(move_actions,seconds_left)
    

    def reduce_boards(self,max_num,delete = False, get_weighting = True):
        if not get_weighting and max_num > self.board_dict.size():
            return 
        possible_boards = self.board_dict.get_boards()
        weights,distances,_ = self.siamese.get_board_weightings(possible_boards)
        weighted_boards = sorted(zip(possible_boards,weights),key = lambda tup:tup[1],reverse = True)
        if delete:
            if self.board_dict.size() < max_num:
                return weighted_boards
            print(f'{self.board_dict.size()} boards before deletion')
            for board,_ in weighted_boards[max_num:]:
                self.board_dict.delete_board(board)
            print(f'{self.board_dict.size()} boards after deletion')
        return weighted_boards[:max_num]


    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        self.siamese.handle_move_result(requested_move,taken_move,captured_opponent_piece,capture_square)
        len_boards_before = self.board_dict.size()
        # handle result of own move, adapt possible board states
        resulting_boards = BoardDict()
        if taken_move is not None:
            if not captured_opponent_piece:
                print("Move did not capture a piece.")
                for board in self.board_dict.get_boards():
                    # if we did not capture a piece, we only keep boards
                    # where there was no piece on the square we moved to.
                    if board.piece_at(taken_move.to_square) is None and \
                            (board.is_pseudo_legal(taken_move) or board.is_castling(taken_move)):
                        new_board = board.copy()
                        new_board.push(taken_move)
                        resulting_boards.add_board(new_board)
            else:
                print("Move captured a piece.")
                for board in self.board_dict.get_boards():
                    # if we captured a piece, we only keep boards
                    # where there was a opponent piece on the square we moved to.
                    if board.piece_at(capture_square) is not None and \
                            board.piece_at(capture_square).color is not self.color and \
                            board.is_pseudo_legal(taken_move) and \
                            capture_square is not board.king(not self.color):
                        new_board = board.copy()
                        new_board.push(taken_move)
                        resulting_boards.add_board(new_board)

        # in the case the requested move was not possible
        elif requested_move != taken_move and taken_move is None:
            print("Move was rejected.")
            # if the actual move was different than the one we took (our move was not possible),
            # we only keep those boards where the requested move is not possible.
            for board in self.board_dict.get_boards():
                # we took a move, so its our turn for all boards
                new_board = board.copy()
                # new_board.turn = not self.color
                if not board.is_legal(requested_move):
                    new_board.push(chess.Move.null())
                    resulting_boards.add_board(new_board)
        # in the case we did not make a move
        else:
            for board in self.board_dict.get_boards():
                new_board = board.copy()
                new_board.push(chess.Move.null())
                resulting_boards.add_board(new_board)

        print(f"Own Move against {self.opponent_name}. Boards before handle_move_result: {len_boards_before}, after: {resulting_boards.size()}")
        if resulting_boards.size() < 1:
            print('Not possible boards, this should not happen')
        # Fixme
        # if we have 0 boards, then the king must have been captured. pls check
        # assert len(resulting_boards) >= 1 or capture_square is [board.king(not self.color) for board in self.board_dict.get_boards()]
        self.board_dict = resulting_boards

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
        try:
            del self.siamese
            self.engine.quit()
            self.engine.close()
            del self.engine
            torch.cuda.empty_cache()
            print('Finished game')
        except chess.engine.EngineTerminatedError: 
            pass
