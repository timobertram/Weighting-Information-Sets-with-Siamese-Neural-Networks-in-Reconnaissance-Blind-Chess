import csv
import json
import lzma
import os
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict
from fileinput import filename

import chess
import numpy as np
import pandas as pd
import requests
import torch
import tqdm
from InfosetGenerator import InfosetGen
from ray.util.multiprocessing import Pool
from utils import int_to_row_column, row_column_to_int, sense_location_to_area

from models import Siamese_RBC_dataset

def fill_codes(codes,borders):
    i = 0
    for nSquares in range(1,8):
        for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            if direction == 'N':
                borders[i,nSquares:,:] = 1
            if direction == 'NE':
                borders[i,nSquares:,:-nSquares] = 1
            if direction == 'E':
                borders[i,:,:-nSquares] = 1
            if direction == 'SE':
                borders[i,:-nSquares,:-nSquares] = 1
            if direction == 'S':
                borders[i,:-nSquares,:] = 1
            if direction == 'SW':
                borders[i,:-nSquares,nSquares:] = 1
            if direction == 'W':
                borders[i,:,nSquares:] = 1
            if direction == 'NW':
                borders[i,nSquares:,nSquares:] = 1
            codes[(nSquares,direction)] = i
            i += 1
    for two in ["N","S"]:
        for one in ["E","W"]:
            if two == 'S' and one == 'E':
                borders[i,:-2,:-1] = 1
            if two == 'N' and one == 'E':
                borders[i,2:,:-1] = 1
            if two == 'S' and one == 'W':
                borders[i,:-2,1:] = 1
            if two == 'N' and one == 'W':
                borders[i,2:,1:] = 1
            codes[("knight", two, one)] , i = i , i + 1
    for two in ["E","W"]:
        for one in ["N","S"]:
            if two == 'E' and one == 'N':
                borders[i,1:,:-2] = 1
            if two == 'W' and one == 'S':
                borders[i,:-1,2:] = 1
            if two == 'E' and one == 'S':
                borders[i,:-1,:-2] = 1
            if two == 'W' and one == 'N':
                borders[i,1:,2:] = 1
            codes[("knight", two, one)] , i = i , i + 1
    for move in ["N","NW","NE"]:
        for promote_to in ["Rook","Knight","Bishop"]:
            borders[i,1,:] = 1
            borders[i,6,:] = 1
            codes[("underpromotion", move, promote_to)] , i = i , i + 1
    return codes,borders



def move_to_location(move,board):
    starting_square = chess.SQUARES[chess.parse_square(move[:2])]
    piece = str(board.piece_at(starting_square))
    end_square = chess.SQUARES[chess.parse_square(move[2:4])]
    start_row = 7-chess.square_rank(starting_square)
    end_row = 7-chess.square_rank(end_square)
    start_col = chess.square_file(starting_square)
    end_col = chess.square_file(end_square)
    promotion = move[4] if len(move) == 5 else None

    row_dif = start_row-end_row
    col_dif = start_col-end_col
    directions = {'S': -row_dif if row_dif < 0 else 0,'N': row_dif if row_dif > 0 else 0,
    'E': -col_dif if col_dif < 0 else 0,'W': col_dif if col_dif > 0 else 0}
    if piece == 'N' or piece == 'n':
        for k,v in directions.items():
            if v == 2:
                first_coordinate = k
            elif v == 1:
                second_coordinate = k
        return codes['knight',first_coordinate,second_coordinate],start_row,start_col
    else:
        if promotion and promotion != 'q':
            done_directions = [d for d in directions.keys() if directions[d] > 0]
            if 'E' in done_directions:
                dir = 'NE'
            elif 'W' in done_directions:
                dir = 'NW'
            else:
                dir = 'N'

            if promotion == 'n':
                promote_to = 'Knight'
            elif promotion == 'r':
                promote_to = 'Rook'
            elif promotion == 'b':
                promote_to = 'Bishop'
            return codes["underpromotion",dir,promote_to],start_row,start_col
        else:
            direction = ''
            move_length = None
            if directions['N'] > 0:
                direction += 'N'
                move_length = directions['N']
            if directions['S'] > 0:
                direction += 'S'
                assert not move_length or move_length == directions['S']
                move_length = directions['S']
            if directions['E'] > 0:
                direction += 'E'
                assert not move_length or move_length == directions['E']
                move_length = directions['E']
            if directions['W'] > 0:
                direction += 'W'
                assert not move_length or move_length == directions['W']
                move_length = directions['W']
            return codes[(move_length,direction)],start_row,start_col

def location_to_move(location, print_out = False):
    try:
        move_type = inverse_codes[location[0]]
    except Exception as e:
        print(inverse_codes)
        print(location[0])
        print(inverse_codes[location[0]])
        print(type(e))
        print(location)
        print(f'Exception with move type')
        raise Exception
    start_col = location[2]
    end_col = start_col
    start_row = location[1]
    end_row = start_row
    start_square = col_to_int[start_col]+row_to_int[start_row]
    piece_promotion = ''
    if move_type[0] == 'knight':
        if move_type[1] == 'S':
            if move_type[2] == 'E':
                end_col = start_col + 1
                end_row = start_row + 2
            else:
                end_col = start_col - 1
                end_row = start_row + 2
        if move_type[1] == 'N':
            if move_type[2] == 'E':
                end_col = start_col + 1
                end_row = start_row - 2
            else:
                end_col = start_col - 1
                end_row = start_row - 2
        if move_type[1] == 'E':
            if move_type[2] == 'N':
                end_col = start_col + 2
                end_row = start_row - 1
            else:
                end_col = start_col + 2
                end_row = start_row + 1
        if move_type[1] == 'W':
            if move_type[2] == 'N':
                end_col = start_col - 2
                end_row = start_row - 1
            else:
                end_col = start_col - 2
                end_row = start_row + 1
    else:
        move_length = move_type[0]
        if move_length == 'underpromotion':
            if move_type[2] == 'Rook':
                piece_promotion = 'r'
            elif move_type[2] == 'Bishop':
                piece_promotion = 'b'
            elif move_type[2] == 'Knight':
                piece_promotion = 'n'
            else:
                print(move_type[2])
                raise Exception
            if start_row == 6:
                end_row = 7
            else:
                end_row = 0
            if 'W' in move_type[1]:
                if start_row == 6:
                    end_col = start_col -1
                else:
                    end_col = start_col - 1
            if 'E' in move_type[1]:
                if start_row == 6:
                    end_col = start_col +1
                else:
                    end_col = start_col + 1
        else:
            if 'N' in move_type[1]:
                end_row = start_row - move_length
            if 'S' in move_type[1]:
                end_row = start_row + move_length
            if 'E' in move_type[1]:
                end_col = start_col + move_length
            if 'W' in move_type[1]:
                end_col = start_col - move_length
    try:
        end_square = col_to_int[end_col]+row_to_int[end_row]
    except Exception as e:
        print(location)
        print(e)
        print(type(e), end_col, end_row, move_type)
        print(f'Move out of board; start {start_square}, {move_type}, start/end col: {start_col}/{end_col}, start/end row: {start_row}/{end_row}')
        return None
    move = start_square+end_square
    if print_out:
        print(f'No error with move {move}, {move_type}, start/end col: {col_to_int[start_col]}/{col_to_int[end_col]}, start/end row: {row_to_int[start_row]}/{row_to_int[end_row]}')
    return move+piece_promotion



def get_all_observations(data):
    len_one_stack = 90
    white_input_history = []
    black_input_history = []

    white_presense_observations = []
    white_premove_observations = []
    black_presense_observations = []
    black_premove_observations = []
    
    white_sense_results = data['sense_results']['true']
    white_player = data['white_name']
    white_senses = data['senses']['true']
    white_capture_squares = data['capture_squares']['true']
    white_requested_moves = data['requested_moves']['true']
    white_taken_moves = data['taken_moves']['true']
    white_fens_before = data['fens_before_move']['true']

    
    black_sense_results = data['sense_results']['false']
    black_player = data['black_name']
    black_senses = data['senses']['false']
    black_capture_squares = data['capture_squares']['false']
    black_requested_moves = data['requested_moves']['false']
    black_taken_moves = data['taken_moves']['false']
    black_fens_before = data['fens_before_move']['false']
    for i,(sense,result,_) in enumerate(zip(white_senses,white_sense_results,white_capture_squares)):     
        target_result = torch.ones(1)
        if not data['winner_color']:
            target_result *= -1      
            won_or_not = 'loss'  
        else:  
            won_or_not = 'won'
        #0: opponents capture
        #1-73: last requested/taken move
        #74: last move captured a piece
        #75: last move was None
        #76-81: own pieces
        #82: sensed area
        #83-88: sensed result
        #last of stack: color
        board = torch.zeros(len_one_stack,8,8)
        board[-1,:,:] = 1

        #fill in last opponent capture if exists
        if i > 0:
            if black_capture_squares[i-1] is not None:
                row,col = int_to_row_column(black_capture_squares[i-1])
                board[pos_opp_capture,row,col] += 1

        #fill in last taken and requested moves
        if i > 0:

            requested_move = white_requested_moves[i-1]['value'] if white_requested_moves[i-1] else None
            if requested_move:
                loc = move_to_location(requested_move,own_pieces)
                board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        #fill in whether last move captured a piece
        if i > 0:
            if white_capture_squares[i-1] is not None:
                row,col = int_to_row_column(white_capture_squares[i-1])
                board[pos_last_move_captured,row,col] += 1

        #fill in whether last move returned None
        if i > 0:
            if white_taken_moves[i-1] is None:
                board[pos_last_move_None,:,:] = 1
            
        #fill in own board
        own_pieces = chess.Board(white_fens_before[i])
        target_pieces = torch.zeros(6)
        for square,piece in own_pieces.piece_map().items():
            if piece.color:
                row,col = int_to_row_column(square)
                board[pos_own_pieces+piece_map[str(piece)][1],row,col]+= 1
            else:
                target_pieces[piece.piece_type-1] += 1
        
        #add sense training data
        white_input_history.append(board)
        target_sense = torch.zeros(8,8)
        if sense is not None:
            sense_row,sense_col = int_to_row_column(sense)
            target_sense[sense_row,sense_col] += 1
        white_presense_observations.append(torch.clone(torch.stack(white_input_history[-20:])))
        
        #fill in sensed area     
        if sense is not None:           
            board[pos_sensed] += sense_location_to_area(sense)

        #fill in sensed pieces
        for res in result:
            if res[1] is not None:
                c,pos = piece_map[res[1]['value']]
                if not c:
                    row,column = int_to_row_column(res[0])
                    board[pos_sense_result+pos,row,column] += 1
        
        tmp_move = torch.zeros(73,8,8)
        done_move = white_requested_moves[i]['value'] if white_requested_moves[i] else None
        target_move = torch.zeros(73*8*8+1)
        if done_move:
            if done_move[-1] == 'q':
                done_move = done_move[:-1]
            loc = move_to_location(done_move,own_pieces)
            tmp_move[loc[0],loc[1],loc[2]] = 1     
            target_move[:-1] = tmp_move.view(-1)
        else:
            continue
        white_input_history[-1] = board
        white_premove_observations.append(torch.clone(torch.stack(white_input_history[-20:])))


    for i,(sense,result,_) in enumerate(zip(black_senses,black_sense_results,black_capture_squares)):
        target_result = torch.ones(1)
        if data['winner_color']:
            target_result *= -1
            won_or_not = 'loss'
        else:
            won_or_not = 'win'
        board = torch.zeros(len_one_stack,8,8)

        #fill in last opponent capture if exists
        if i > 0:
            if white_capture_squares[i-1] is not None:
                row,col = int_to_row_column(white_capture_squares[i-1])
                board[pos_opp_capture,row,col] += 1

        #fill in last taken and requested moves
        if i > 0:

            requested_move = black_requested_moves[i-1]['value'] if black_requested_moves[i-1] else None
            if requested_move:
                loc = move_to_location(requested_move,own_pieces)
                board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        #fill in whether last move captured a piece
        if i > 0:
            if black_capture_squares[i-1] is not None:
                row,col = int_to_row_column(black_capture_squares[i-1])
                board[pos_last_move_captured,row,col] += 1

        #fill in whether last move returned None
        if i > 0:
            if black_taken_moves[i-1] is None:
                board[pos_last_move_None,:,:] = 1
        
        #fill in own board
        own_pieces = chess.Board(black_fens_before[i])
        target_pieces = torch.zeros(6)
        for square,piece in own_pieces.piece_map().items():
            if not piece.color:
                row,col = int_to_row_column(square)
                board[pos_own_pieces+piece_map[str(piece)][1],row,col]+= 1
            else:
                target_pieces[piece.piece_type-1] += 1
        
        #add sense training data
        black_input_history.append(board)    

        
        target_sense = torch.zeros(8,8)
        if sense is not None:
            sense_row,sense_col = int_to_row_column(sense)
            target_sense[sense_row,sense_col] += 1  
        black_presense_observations.append(torch.clone(torch.stack(black_input_history[-20:])))
        
        #fill in sensed area     
        if sense is not None:           
            board[pos_sensed] += sense_location_to_area(sense)

        #fill in sensed pieces
        for res in result:
            if res[1] is not None:
                c,pos = piece_map[res[1]['value']]
                if c:
                    row,column = int_to_row_column(res[0])
                    board[pos_sense_result+pos,row,column] += 1
        
        tmp_move = torch.zeros(73,8,8)
        done_move = black_requested_moves[i]['value'] if black_requested_moves[i] else None
        target_move = torch.zeros(73*8*8+1)
        if done_move:
            if done_move[-1] == 'q':
                done_move = done_move[:-1]
            loc = move_to_location(done_move,own_pieces)
            tmp_move[loc[0],loc[1],loc[2]] = 1     
            target_move[:-1] = tmp_move.view(-1)
        else:
            continue
        black_input_history[-1] = board
        black_premove_observations.append(torch.clone(torch.stack(black_input_history[-20:])))
    return white_presense_observations,white_premove_observations,black_presense_observations,black_premove_observations



def get_all_white_boards(data):
    sense_boards = []
    move_boards = []
    true_boards = []
    start_time = time.time()
    all_seconds = 900
    agent = InfosetGen()
    agent.handle_game_start(True, chess.Board(data['fens_before_move']['true'][0]),data['black_name'])
    agent.handle_opponent_move_result(None, None)
    for turn in range(len(data['senses']['true'])):
        sense_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        agent.handle_sense_result(data['sense_results']['true'][turn])

        if turn >= len(data['fens_before_move']['true']):
            break

        true_boards.append(data['fens_before_move']['true'][turn])
        if agent.board_dict.size() == 0:
            break
        move_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        requested_move = chess.Move.from_uci(data['requested_moves']['true'][turn]['value']) if data['requested_moves']['true'][turn] else None
        taken_move = chess.Move.from_uci(data['taken_moves']['true'][turn]['value']) if data['taken_moves']['true'][turn] else None
        agent.handle_move_result(requested_move,taken_move,\
            True if data['capture_squares']['true'][turn] != None else False, data['capture_squares']['true'][turn])
        if agent.board_dict.size() == 0:
            break
        if len(data['capture_squares']['false']) > turn:
            agent.handle_opponent_move_result(True if data['capture_squares']['false'][turn] != None else False, data['capture_squares']['false'][turn])
        
            if agent.board_dict.size() == 0:
                break
    
    #agent.handle_opponent_move_result(False, None)
    del data
    return sense_boards,move_boards,true_boards,agent

def get_all_black_boards(data):
    sense_boards = []
    move_boards = []
    true_boards = []
    all_seconds = 900
    start_time = time.time()

    agent = InfosetGen()
    agent.handle_game_start(False, chess.Board(data['fens_before_move']['false'][0]),data['white_name'])
    
    for turn in range(len(data['taken_moves']['false'])):
        true_boards.append(data['fens_before_move']['false'][turn])
        agent.handle_opponent_move_result(True if data['capture_squares']['true'][turn] != None else False, data['capture_squares']['true'][turn])
        
        if agent.board_dict.size() == 0:
            break
        sense_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        agent.handle_sense_result(data['sense_results']['false'][turn])

        if agent.board_dict.size() == 0:
            break
        move_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        agent.choose_move(list(chess.Board(data['fens_before_move']['false'][turn]).pseudo_legal_moves), all_seconds-(time.time()-start_time))
        requested_move = chess.Move.from_uci(data['requested_moves']['false'][turn]['value']) if data['requested_moves']['false'][turn] else None
        taken_move = chess.Move.from_uci(data['taken_moves']['false'][turn]['value']) if data['taken_moves']['false'][turn] else None
        agent.handle_move_result(requested_move,taken_move,\
            True if data['capture_squares']['false'][turn] != None else False, data['capture_squares']['false'][turn])

        if agent.board_dict.size() == 0:
            break
    
    if len(data['taken_moves']['false']) > len(data['senses']['false']):
        agent.handle_sense_result(data['sense_results']['false'][-1])
        agent.choose_move(list(chess.Board(data['fens_before_move']['false'][-1]).pseudo_legal_moves), 1000)

    del data
    return sense_boards,move_boards,true_boards,agent

def process_files(file):
    finished_path = outpath+'finished_files/'

    with open(path+file) as f:
        data = json.load(f)
        if len(data['fens_before_move']['true']) == 0 or len(data['fens_before_move']['false']) == 0:
            return

    os.makedirs(finished_path, exist_ok=True)
    if os.path.exists(finished_path+file):
        print(f'Skipping file {file} as it was already processed')
        return

    if not os.path.exists(outpath+'/move/'):
        os.mkdir(outpath+'/move/')
    if not os.path.exists(outpath+'/sense/'):
        os.mkdir(outpath+'/sense/')
    white_name = data['white_name']
    black_name = data['black_name']
    white_presense_observations,white_premove_observations,black_presense_observations,black_premove_observations = get_all_observations(data)
    white_sense_boards,white_move_boards,white_true_boards,_ = get_all_white_boards(data)
    black_sense_boards,black_move_boards,black_true_boards,_ = get_all_black_boards(data)
    white_sense_num = min(len(white_sense_boards),len(white_true_boards),len(white_presense_observations))
    white_move_num = min(len(white_move_boards),len(white_true_boards),len(white_premove_observations))
    black_sense_num = min(len(black_sense_boards),len(black_true_boards),len(black_presense_observations))
    black_move_num = min(len(black_move_boards),len(black_true_boards),len(black_premove_observations))



    for i,(boards,true_board,obs) in enumerate(zip(white_sense_boards[1:white_sense_num], white_true_boards[1:white_sense_num],white_presense_observations[1:white_sense_num])):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/sense/{file if file else online}_white_{i}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards, black_name), f)
                f.close()

    for i,(boards,true_board,obs) in enumerate(zip(white_move_boards[1:white_move_num], white_true_boards[1:white_move_num],white_premove_observations[1:white_move_num])):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/move/{file if file else online}_white_{i}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards, black_name), f)
                f.close()

    for i,(boards,true_board,obs) in enumerate(zip(black_sense_boards[:black_sense_num], black_true_boards[:black_sense_num],black_presense_observations[:black_sense_num])):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/sense/{file if file else online}_black_{i}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards, white_name), f)
                f.close()
    for i,(boards,true_board,obs) in enumerate(zip(black_move_boards[:black_move_num], black_true_boards[:black_move_num],black_premove_observations[:black_move_num])):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/move/{file if file else online}_black_{i}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards,white_name), f)
                f.close()
    if file is not None:
        open(finished_path+file, 'w')
    del file, data, white_sense_boards,white_true_boards,white_presense_observations,white_premove_observations,black_sense_boards,black_true_boards,black_presense_observations,black_premove_observations



def split_data(path):
    for fold in os.listdir(path):
        if fold == 'finished_files':
            continue
        
        train_path = path+'train/'+fold+'/'
        test_path = path+'val/'+fold+'/'
        os.makedirs(name=train_path,exist_ok=True)
        os.makedirs(name=test_path,exist_ok=True)


        files = os.listdir(path+fold)
        len_test = len(files)//10
        test_files = np.random.choice(files,len_test)
        for file in files:
            if not '.pt' in file:
                continue
            if file in test_files:
                os.rename(f'{path}{fold}/{file}',test_path+file)
            else:
                os.rename(f'{path}{fold}/{file}',train_path+file)
        os.removedirs(name=path+fold)

# positions of the features in the board representation
pos_opp_capture = 0
pos_last_moves = 1
pos_last_move_captured = 74
pos_last_move_None = 75
pos_own_pieces = 76
pos_sensed = 82
pos_sense_result = 83
piece_map = {'p': (0,0),'r': (0,1),'n': (0,2),'b': (0,3),'q': (0,4),'k': (0,5),'P': (1,0),
    'R': (1,1),'N': (1,2),'B': (1,3),'Q': (1,4),'K': (1,5)}

codes = {}
borders = torch.zeros(73, 8,8, dtype= torch.bool)
codes, borders = fill_codes(codes,borders)
inverse_codes = {value:key for (key,value) in codes.items()}
columns = { k:v for v,k in enumerate("abcdefgh")}
piece_map = {'p': (0,0),'r': (0,1),'n': (0,2),'b': (0,3),'q': (0,4),'k': (0,5),'P': (1,0),
    'R': (1,1),'N': (1,2),'B': (1,3),'Q': (1,4),'K': (1,5)}
notation_map = {'a':0, 'b':1,'c':2, 'd':3,'e':4, 'f':5,'g':6, 'h':7,
    '1':7,'2':6,'3':5,'4':4,'5':3,'6':2,'7':1,'8':0,
    1:7,2:6,3:5,4:4,5:3,6:2,7:1,8:0}


if __name__ == '__main__':
    path = 'data/games/'
    outpath = 'data/siamese/'
    files = os.listdir(path)
    pool = Pool()

    results = pool.map(process_files, files)
    pool.close()
    pool.join()

    split_data(outpath)