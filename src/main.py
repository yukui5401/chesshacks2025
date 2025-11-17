from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn.functional as F
import random
import os

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
NEW_MODEL_PATH = os.path.join(os.path.dirname(__file__), "latent_encoder.pt")

from .utils import LatentEncoder, fen_to_tensor

# -------------------------
# Load model
# -------------------------

model = LatentEncoder().to(DEVICE)
checkpoint = torch.load(NEW_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -------------------------
# Board embedding
# -------------------------
def board_embedding(board):
    fen = board.fen()
    x = fen_to_tensor(fen).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z = model(x)
        z = F.normalize(z, p=2, dim=-1)
    return z.squeeze(0)


# -------------------------
# Entrypoint: 5-ply beam search move selector
# -------------------------
@chess_manager.entrypoint
def test_func(ctx: GameContext, depth=3, beam_width=5):
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    current_emb = board_embedding(ctx.board)

    # -------------------------
    # Recursive beam search
    # -------------------------
    def recurse(board, emb, d):
        if d == 0:
            return 0.0, None

        moves = list(board.generate_legal_moves())
        scored_moves = []

        for move in moves:
            board.push(move)
            next_emb = board_embedding(board)
            # recursively search next ply
            child_score, _ = recurse(board, next_emb, d - 1)
            board.pop()
            total_score = torch.dot(emb, next_emb).item() + child_score
            scored_moves.append((move, total_score))

        # beam pruning: keep top-K moves
        top_moves = sorted(scored_moves, key=lambda x: x[1], reverse=True)[:beam_width]
        return top_moves[0][1], top_moves[0][0]

    _, chosen_move = recurse(ctx.board, current_emb, depth)

    # -------------------------
    # Log probabilities (optional)
    # -------------------------
    ctx.logProbabilities({chosen_move: 1.0})

    return chosen_move


# -------------------------
# Reset
# -------------------------
@chess_manager.reset
def reset_func(ctx: GameContext):
    pass
