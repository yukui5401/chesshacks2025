from .utils import chess_manager, GameContext
from chess import Move
import torch
import numpy as np
import random
from .utils import ChessAlphaZeroResNet, fen_to_tensor
import json

# ------------------------
# Device
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load trained model
# ------------------------
ckpt_path = "src/epochs_2.ckpt"  # change to your checkpoint
model = ChessAlphaZeroResNet.load_from_checkpoint(ckpt_path)
model.to(device)
model.eval()  # disables dropout/batchnorm updates

# ------------------------
# Load move_to_index mapping
# ------------------------
with open("src/move_to_index.json", "r") as f:
    move_to_index = json.load(f)
index_to_move = {v: k for k, v in move_to_index.items()}


# ------------------------
# Reset function
# ------------------------
@chess_manager.reset
def reset_func(ctx: GameContext):
    # Called when a new game begins
    # Nothing to reset in this simple stateless model
    pass


# ------------------------
# Entrypoint function
# ------------------------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # Encode current board

    board_tensor = fen_to_tensor(ctx.board.fen())
    board_tensor = (
        torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    )  # add batch dim

    # # Predict policy
    with torch.no_grad():
        policy_logits, _ = model(board_tensor)
        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy().flatten()

    # # Get legal moves and their indices in the policy
    legal_moves = list(ctx.board.generate_legal_moves())
    legal_sans = [ctx.board.san(move) for move in legal_moves]

    legal_indices = []
    filtered_sans = []
    for san, move in zip(legal_sans, legal_moves):
        if san in move_to_index:
            legal_indices.append(move_to_index[san])
            filtered_sans.append(san)

    # If no legal move is in training mapping, fallback to uniform random
    if not legal_indices:
        chosen_move = random.choice(legal_moves)
        ctx.logProbabilities(
            {ctx.board.san(m): 1 / len(legal_moves) for m in legal_moves}
        )
        return chosen_move

    # Get probabilities for only legal moves
    probs = policy_probs[legal_indices]
    probs /= probs.sum()

    # Sample move according to policy
    chosen_san = np.random.choice(filtered_sans, p=probs)
    chosen_move = ctx.board.parse_san(chosen_san)

    return chosen_move
