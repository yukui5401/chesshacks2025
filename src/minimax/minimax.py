from chess import Board, Color
import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


def get_material_count(board: Board, color: Color):
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        squares = board.pieces(piece_type, color)
        total += len(squares) * value
    return total


def minimax(board: Board, depth: int, maximizingPlayer: bool, alpha=float('-inf'), beta=float('inf')):
    if (depth == 0) or (board.is_game_over()):
        score = get_material_count(
            board, chess.WHITE if board.turn else chess.BLACK)
        return score, None

    if maximizingPlayer:
        value = float('-inf')
        possible_moves = board.generate_legal_moves()

        for move in possible_moves:
            board.push(move)
            tmp = minimax(board, depth-1, False, alpha, beta)[0]
            board.pop()

            if tmp > value:
                value = tmp
                best_movement = move

            if value >= beta:
                break

            alpha = max(alpha, value)
    else:
        value = float('inf')
        possible_moves = board.generate_legal_moves()
        for move in possible_moves:
            board.push(move)
            tmp = minimax(board, depth-1, True, alpha, beta)[0]
            board.pop()

            if tmp < value:
                value = tmp
                best_movement = move

            if value <= alpha:
                break

            beta = min(beta, value)

    return value, best_movement
