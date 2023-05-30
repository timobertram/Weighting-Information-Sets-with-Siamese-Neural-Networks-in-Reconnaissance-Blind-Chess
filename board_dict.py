import copy
from typing import List, Tuple, Optional, Dict

import chess.engine
from chess import Square
import time


class BoardDict:
    def __init__(self):
        self.__refs_to_boards: Dict[int, chess.Board] = {}

    def add_board(self, board: chess.Board):
        self.__refs_to_boards[id(board)] = board

    def get_boards(self):
        return list(self.__refs_to_boards.values())

    def size(self):
        return len(self.__refs_to_boards)

    def delete_boards(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        """
        This function deletes boards that do not fit 'sense_result'
        """
        deleted_boards = []

        for board in self.__refs_to_boards.values():
            for res in sense_result:
                if type(res[1]) == chess.Piece:
                    if (str(board.piece_at(res[0])) if board.piece_at(res[0]) else None) != (str(res[1]) if res[1] else None):
                        deleted_boards.append(board)
                        break
                else:
                    if (str(board.piece_at(res[0])) if board.piece_at(res[0]) else None) != (str(res[1]['value']) if res[1] else None):
                        deleted_boards.append(board)
                        break

        for board in deleted_boards:
            self.__refs_to_boards.pop(id(board), None)

    def delete_board(self, board: chess.Board):
        """
        This function deletes the board specified by 'board' from the BoardDict
        """
        self.__refs_to_boards.pop(id(board), None)


if __name__ == "__main__":
    bd: BoardDict = BoardDict()
    b = chess.Board()

    # create board samples
    start: float = time.time()
    boards = [b]
    from_idx = 0
    for l in range(4):
        to_idx = len(boards)
        for i in range(from_idx, to_idx):
            board = boards[i]
            for move in board.pseudo_legal_moves:
                new_board = copy.deepcopy(board)
                new_board.push(move)
                boards.append(new_board)
        from_idx = to_idx
    board_count = len(boards)
    print(f'number of boards: {board_count}')
    print(f'time of creating board samples: {time.time() - start}')

    sense_result = [(0, None)]

    # with the class BoardDict
    print('\n\nWith class BoardDict')

    start = time.time()
    for b in boards:
        bd.add_board(b)
    print(f'time of adding boards to BoardDict: {time.time() - start}')
    start = time.time()
    bd.delete_boards(sense_result)
    print(f'number of deleted boards: {board_count - bd.size()}')
    print(f'time of deleting boards: {time.time() - start}')

    # without the index structure
    print('\n\nWithout class BoardDict')

    start: float = time.time()
    deleted_boards = []
    for board in boards:
        for res in sense_result:
            if board.piece_at(res[0]) != res[1]:
                deleted_boards.append(board)
                break
    print(f'time of getting deleted boards: {time.time() - start}')
    start = time.time()
    for board in deleted_boards:
        boards.remove(board)
    print(f'number of deleted boards: {board_count - len(boards)}')
    print(f'time of deleting boards: {time.time() - start}')
