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
NEW_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "encoder_checkpoint_step50.pt"
)

# -------------------------
# Import models
# -------------------------
from .utils import LatentEncoder, fen_to_tensor

# -------------------------
# Load chosen model
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
# Entrypoint: select move
# -------------------------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    current_emb = board_embedding(ctx.board)
    move_scores = []

    for move in legal_moves:
        ctx.board.push(move)
        next_emb = board_embedding(ctx.board)
        ctx.board.pop()
        score = torch.dot(current_emb, next_emb).item()
        move_scores.append(max(score, 0.0))

    if sum(move_scores) == 0:
        move_scores = [1.0] * len(legal_moves)

    total = sum(move_scores)
    move_probs = {move: score / total for move, score in zip(legal_moves, move_scores)}
    ctx.logProbabilities(move_probs)

    chosen_move = random.choices(legal_moves, weights=move_scores, k=1)[0]
    return chosen_move


# -------------------------
# Reset
# -------------------------
@chess_manager.reset
def reset_func(ctx: GameContext):
    pass
