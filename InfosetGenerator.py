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

class InfosetGen(Player):
    def __init__(self):
        self.board_dict = BoardDict()
        self.color = None
        self.first_turn = True

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board_dict.add_board(board)
        self.color = color
        self.opponent_name = opponent_name


    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        max_board_num = 5_000

        current_boards = self.board_dict.get_boards()
        shuffle(current_boards)
        # for some reasons the game tries to notify of opponent turn on first move for white so ignore that
        if self.first_turn and self.color:
            self.first_turn = False
            return

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


    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        self.board_dict.delete_boards(sense_result)


    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        resulting_boards = BoardDict()
        if taken_move is not None:
            if not captured_opponent_piece:
                for board in self.board_dict.get_boards():
                    # if we did not capture a piece, we only keep boards
                    # where there was no piece on the square we moved to.
                    if board.piece_at(taken_move.to_square) is None and \
                            (board.is_pseudo_legal(taken_move) or board.is_castling(taken_move)):
                        new_board = board.copy()
                        new_board.push(taken_move)
                        resulting_boards.add_board(new_board)
            else:
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
        self.board_dict = resulting_boards