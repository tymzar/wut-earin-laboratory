import math
import copy
from enum import Enum


class CellValues(Enum):
    EMPTY = None
    O = "O"
    X = "X"


def get_initial_state():

    return [
        [CellValues.EMPTY.value, CellValues.EMPTY.value, CellValues.EMPTY.value],
        [CellValues.EMPTY.value, CellValues.EMPTY.value, CellValues.EMPTY.value],
        [CellValues.EMPTY.value, CellValues.EMPTY.value, CellValues.EMPTY.value],
    ]


def get_current_player(board):

    amount_of_empty = 0

    for i in range(3):
        amount_of_empty += board[i].count(CellValues.EMPTY.value)

    return CellValues.O.value if amount_of_empty % 2 == 0 else CellValues.X.value


def get_possible_actions(board):
    possible_actions = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == CellValues.EMPTY.value:
                possible_actions.add((i, j))

    return possible_actions


def move(board, action):
    x, y = action
    current_player = get_current_player(board)
    next_board = copy.deepcopy(board)
    next_board[x][y] = current_player

    return next_board


def get_winner(board):

    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != CellValues.EMPTY.value:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != CellValues.EMPTY.value:
            return board[0][i]

    if board[0][0] == board[1][1] == board[2][2] != CellValues.EMPTY.value:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] != CellValues.EMPTY.value:
        return board[0][2]

    return CellValues.EMPTY.value


def is_game_finished(board):

    if get_winner(board) != CellValues.EMPTY.value:
        return True

    for i in range(3):
        if CellValues.EMPTY.value in board[i]:
            return False

    return True


def get_game_status(board):

    game_winner = get_winner(board)

    return (
        1
        if game_winner == CellValues.X.value
        else -1 if game_winner == CellValues.O.value else 0
    )


def minimax(board):

    current_player = get_current_player(board)

    optimal_move = CellValues.EMPTY.value

    if is_game_finished(board):
        return optimal_move

    beta = math.inf
    alpha = -math.inf

    if current_player == CellValues.X.value:
        best_score = -math.inf
        for action in get_possible_actions(board):
            score = min_value(move(board, action), alpha, beta)
            if score > best_score:
                best_score = score
                optimal_move = action
            if score == 1:
                return optimal_move
    else:
        best_score = math.inf
        for action in get_possible_actions(board):
            score = max_value(move(board, action), alpha, beta)
            if score < best_score:
                best_score = score
                optimal_move = action
            if score == -1:
                return optimal_move

    return optimal_move


def max_value(board, alpha, beta):

    if is_game_finished(board):
        return get_game_status(board)

    max_value = -math.inf

    for action in get_possible_actions(board):
        max_value = max(max_value, min_value(move(board, action), alpha, beta))
        if max_value >= beta:
            return max_value
        alpha = max(alpha, max_value)
    return max_value


def min_value(board, alpha, beta):

    if is_game_finished(board):
        return get_game_status(board)

    min_value = math.inf

    for action in get_possible_actions(board):
        min_value = min(min_value, max_value(move(board, action), alpha, beta))
        if min_value <= alpha:
            return min_value
        beta = min(beta, min_value)
    return min_value
