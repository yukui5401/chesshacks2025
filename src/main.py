from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn.functional as F
import random
import time
from .utils import FENTokenizer, LatentTransformer
import os

# -------------------------
# Config (match training)
# -------------------------
SEQ_LEN = 77
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
DROPOUT = 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model_step16000.pt")

# -------------------------
# Load model and tokenizer
# -------------------------
tokenizer = FENTokenizer(seq_len=SEQ_LEN)
model = LatentTransformer(
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    vocab_size=tokenizer.vocab_size,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()
print("LatentTransformer loaded on", DEVICE)


# -------------------------
# Get board embedding
# -------------------------
def board_embedding(board):
    fen = board.fen()
    tokens = tokenizer.fen_to_tokens(fen)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)  # batch=1
    with torch.no_grad():
        z = model(tokens)
        z = F.normalize(z, p=2, dim=-1)
    return z.squeeze(0)  # [128]


# -------------------------
# Entrypoint: select move
# -------------------------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Current board embedding
    current_emb = board_embedding(ctx.board)
    move_scores = []

    for move in legal_moves:
        ctx.board.push(move)
        next_emb = board_embedding(ctx.board)
        ctx.board.pop()

        # Score = dot product with current embedding
        score = torch.dot(current_emb, next_emb).item()
        move_scores.append(max(score, 0.0))  # ensure non-negative

    # Fallback if all scores are zero
    if sum(move_scores) == 0:
        move_scores = [1.0] * len(legal_moves)

    total = sum(move_scores)
    move_probs = {move: score / total for move, score in zip(legal_moves, move_scores)}
    ctx.logProbabilities(move_probs)

    # Sample move according to scores
    chosen_move = random.choices(legal_moves, weights=move_scores, k=1)[0]
    return chosen_move


# -------------------------
# Reset function
# -------------------------
@chess_manager.reset
def reset_func(ctx: GameContext):
    pass
