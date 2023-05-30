import torch
import chess
import chess.engine
import numpy as np
import os
import time

piece_to_index= {
    'P':0, 'R':1,'N':2,'B':3,'Q':4,'K':5,
    'p':6, 'r':7,'n':8, 'b': 9,'q':10,'k':11
}

index_to_piece = {v:k for k,v in piece_to_index.items()}


codes = {}
borders = torch.zeros(73, 8,8, dtype= torch.bool)



def sense_location_to_area(location):
    area = torch.zeros(8,8)
    row,column = int_to_row_column(location)

    for r in [-1,0,1]:
        for c in [-1,0,1]:
            if row+r < 0 or row+r > 7 or column+c < 0 or column+c > 7:
                continue
            area[row+r,column+c] += 1
    return area
    
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


codes, borders = fill_codes(codes,borders)

def last_move_to_capture_square(move):
    return(int_to_row_column(move.to_square))


def int_to_row_column(location):
    return 7-location//8,location%8

def row_column_to_int(row,column):
    return (7-row)*8+column

def board_from_fen(board):
    tensor_board = torch.zeros(12,8,8)
    board = chess.Board(board)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor_board[piece_to_index[str(piece)],7-chess.square_rank(square),chess.square_file(square)] = 1
    return tensor_board

def board_from_tensor(tensor, color = 1):
    new_board = chess.Board(fen = None)
    for i in range(12):
        current_piece = index_to_piece[i]
        for row in range(8):
            for column in range(8):
                if tensor[i,row,column]:
                    new_board.set_piece_at(row_column_to_int(row,column),chess.Piece.from_symbol(current_piece))
    new_board.turn = color
    return new_board

def create_engine():
    STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
        
    if STOCKFISH_ENV_VAR not in os.environ:
        os.environ['STOCKFISH_EXECUTABLE'] = "/home/fawler/tbertram/RBC/stockfish_14.1_linux_x64_ssse"
    stockfish_path = os.environ[STOCKFISH_ENV_VAR]
    if not os.path.exists(stockfish_path):
        raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))
    return chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)

def get_best_move(board, engine = None, time = 0.001, depth = None):
    if not engine:
        engine = create_engine()
    turn = board.turn
    if board.attackers(turn, board.king(not turn)):
        return list(board.attackers(turn, board.king(not turn)))[0]
    elif engine.analyse(board,chess.engine.Limit(time = 0.001))['score'].relative.is_mate() == 0 and board.attackers(not turn, board.king(turn)):
        return None
    try:
        if depth:
            best_move = engine.play(board,chess.engine.Limit(depth = depth))
        else:
            best_move = engine.play(board,chess.engine.Limit(time = time))
    except:
        print(board)
        raise Exception('Stockfish failed to evaluate')
    return best_move.move

def stockfish_eval(fen = None, board = None, engine = None, time = 0.0001):
    if board is None:
        board = chess.Board(fen)
    turn = board.turn
    
    # if there are any ally pieces that can take king, execute one of those moves
    if board.attackers(turn, board.king(not turn)):
        return 1
    else:
        if not engine:
            engine = create_engine()
        try:
            eval = engine.analyse(board,chess.engine.Limit(time = 0.001))['score'].relative
        except:
            print(board)
            raise Exception('Stockfish failed to evaluate')
        if eval.is_mate():
            if eval.mate() > 0:
                return 1 if turn else 0 
            elif eval.mate() < 0:
                return 0 if turn else 1
            elif eval.mate() == 0 and board.attackers(not turn, board.king(turn)):
                return 0 
        score = eval.score()
        if score == None:
            print(board.fen(), board)
    norm_score = min(1,max((1 / (1 + np.exp(-2 * np.log(3) / 3_000 * score)) + 0.02),0))
    if norm_score < 0:
        print('Score below 0')
        print(eval)
        print(score)
    return norm_score

def illegal_castling(board, color):
    if color:
        castling_boards = []
        splits = board.fen().split('/')
        bP_split = splits[0]
        c_allowed_check = splits[-1]
        c_allowed_check = c_allowed_check.split(' ')
        c_allowed_check = c_allowed_check[-4]
        # black King_Side Caslting
        if "k" in c_allowed_check:
            if bP_split[-3:] == "k2r":
                lst = list(bP_split)
                lst[-1] = "1"
                lst[-2] = "k"
                lst[-3] = "r"
                if board.piece_at(59) is not None:
                    lst.insert(-3, '1')
                else:
                    if board.piece_at(58) is not None:
                        lst[-4] = "2"
                    else:
                        if board.piece_at(57) is not None:
                            lst[-4] = "3"
                        else:
                            if board.piece_at(56) is not None:
                                lst[-4] = "4"
                            else:
                                lst[-4] = "5"
                bP_split = "".join(lst)
                splits[0] = bP_split
                bP_fen = splits[-1]
                bP_fen_split = bP_fen.split(' ')
                bP_fen_split[1] = 'w'
                n = int(bP_fen_split[-2])
                n += 1
                bP_fen_split[-2] = str(n)
                n = int(bP_fen_split[-1])
                n += 1
                bP_fen_split[-1] = str(n)
                KQkq = bP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.islower())
                if len(KQkq) == 0:
                    KQkq = "-"
                bP_fen_split[2] = KQkq
                bP_fen = " ".join(bP_fen_split)
                splits[-1] = bP_fen
                fen_of_needed_board = "/".join(splits)
                new_b = chess.Board(fen_of_needed_board)
                castling_boards.append(new_b)
        # black Queen_Side Caslting
        if "q" in c_allowed_check:
            if bP_split[:3] == "r3k":
                lst = list(bP_split)
                lst[0] = "2"
                lst[1] = "k"
                lst[2] = "r"
                if board.piece_at(61) is not None:
                    lst.insert(3, '1')
                else:
                    if board.piece_at(62) is not None:
                        lst[3] = "2"
                    else:
                        if board.piece_at(63) is not None:
                            lst[3] = "3"
                        else:
                            lst[3] = "4"

                bP_split = "".join(lst)
                splits[0] = bP_split
                bP_fen = splits[-1]
                bP_fen_split = bP_fen.split(' ')
                bP_fen_split[1] = 'w'
                n = int(bP_fen_split[-2])
                n += 1
                bP_fen_split[-2] = str(n)
                n = int(bP_fen_split[-1])
                n += 1
                bP_fen_split[-1] = str(n)
                KQkq = bP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.islower())
                if len(KQkq) == 0:
                    KQkq = "-"
                bP_fen_split[2] = KQkq
                bP_fen = " ".join(bP_fen_split)
                splits[-1] = bP_fen
                fen_of_needed_board = "/".join(splits)
                new_b = chess.Board(fen_of_needed_board)
                castling_boards.append(new_b)
    else:
        castling_boards = []
        wP_fen_splits = board.fen().split('/')
        wP_fen = wP_fen_splits[-1]
        wP_fen_split = wP_fen.split(' ')
        wp_real_fen = wP_fen_split[0]
        c_allowed_check = wP_fen_split[-4]
        # white King-Side Caslting
        if "K" in c_allowed_check:
            if wp_real_fen[-3:] == "K2R":
                lst = list(wp_real_fen)
                lst[-1] = "1"
                lst[-2] = "K"
                lst[-3] = "R"
                if board.piece_at(3) is not None:
                    lst.insert(-3, '1')
                else:
                    if board.piece_at(2) is not None:
                        lst[-4] = "2"
                    else:
                        if board.piece_at(1) is not None:
                            lst[-4] = "3"
                        else:
                            if board.piece_at(0) is not None:
                                lst[-4] = "4"
                            else:
                                lst[-4] = "5"
                wp_fen = "".join(lst)
                wP_fen_split[0] = wp_fen
                wP_fen_split[1] = 'b'
                n = int(wP_fen_split[-2])
                n += 1
                wP_fen_split[-2] = str(n)
                KQkq = wP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.isupper())
                if len(KQkq) == 0:
                    KQkq = "-"
                wP_fen_split[2] = KQkq
                wP_fen = " ".join(wP_fen_split)
                wP_fen_splits[-1] = wP_fen
                fen_of_needed_board = "/".join(wP_fen_splits)
                new_b = chess.Board(fen_of_needed_board)
                castling_boards.append(new_b)
            # white Queen-Side Caslting
        if "Q" in c_allowed_check:
            if wp_real_fen[:3] == "R3K":
                lst = list(wp_real_fen)
                lst[0] = "2"
                lst[1] = "K"
                lst[2] = "R"
                if board.piece_at(5) is not None:
                    lst.insert(3, '1')
                else:
                    if board.piece_at(6) is not None:
                        lst[3] = "2"
                    else:
                        if board.piece_at(7) is not None:
                            lst[3] = "3"
                        else:
                            lst[3] = "4"
                wp_fen = "".join(lst)
                wP_fen_split[0] = wp_fen
                wP_fen_split[1] = 'b'
                n = int(wP_fen_split[-2])
                n += 1
                wP_fen_split[-2] = str(n)
                KQkq = wP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.isupper())
                if len(KQkq) == 0:
                    KQkq = "-"
                wP_fen_split[2] = KQkq
                wP_fen = " ".join(wP_fen_split)
                wP_fen_splits[-1] = wP_fen
                fen_of_needed_board = "/".join(wP_fen_splits)
                new_b = chess.Board(fen_of_needed_board)
                castling_boards.append(new_b)
    return castling_boards

            
def square_to_row_column(square):
    return square//8,square%8

def row_column_to_square(row,column):
    return row*8+column

def get_adjacent_squares(square):
    adjacent_squares = []
    row,column = square_to_row_column(square)
    for i in range(-1,2):
        for j in range(-1,2):
            new_row = row+i
            new_column = column+j
            if 0 <= new_row < 8 and 0 <= new_column < 8:
                adjacent_squares.append(row_column_to_square(new_row,new_column))
    return adjacent_squares

def sense_result_to_string(board,squares,relevant_squares):
    result = ''
    for square in squares:
        if square in relevant_squares:
            res = board.piece_at(square)
            if res is not None:
                result += str(res)
            else:
                result += '0'
        else:
            result += 'x'
    return result


def reduce_fen(fen):
    return fen.split('-')[0]

    
def board_from_fen(board):
    tensor_board = torch.zeros(12,8,8)
    board = chess.Board(board)
    for square,piece in board.piece_map().items():
        tensor_board[piece_to_index[str(piece)],7-chess.square_rank(square),chess.square_file(square)] = 1
    return tensor_board

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


def sense_location_to_area(location):
    area = torch.zeros(8,8)
    row,column = int_to_row_column(location)

    for r in [-1,0,1]:
        for c in [-1,0,1]:
            if row+r < 0 or row+r > 7 or column+c < 0 or column+c > 7:
                continue
            area[row+r,column+c] += 1
    return area