import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from typing import List, Dict


# -------------------------
# FEN Tokenizer
# -------------------------
class FENTokenizer:
    def __init__(self, seq_len: int = 77):
        self.seq_len = seq_len
        self.square_base = 0
        self.square_vocab = 13
        self.side_base = self.square_base + self.square_vocab
        self.side_vocab = 2
        self.castle_base = self.side_base + self.side_vocab
        self.castle_vocab = 2
        self.ep_base = self.castle_base + 4 * self.castle_vocab
        self.ep_vocab = 9
        self.halfmove_base = self.ep_base + self.ep_vocab
        self.halfmove_vocab = 16
        self.fullmove_base = self.halfmove_base + self.halfmove_vocab
        self.fullmove_vocab = 16
        self.total_used = 72
        self.vocab_size = self.fullmove_base + self.fullmove_vocab
        self.pad_token = self.vocab_size
        self.vocab_size += 1
        if self.seq_len < 72:
            raise ValueError("seq_len must be >= 72")
        self.pad_len = self.seq_len - 72

    def fen_to_tokens(self, fen: str) -> List[int]:
        parts = fen.strip().split()
        placement = parts[0].split("/")
        square_tokens = []
        for rank in placement:
            for ch in rank:
                if ch.isdigit():
                    square_tokens.extend([0] * int(ch))
                else:
                    square_tokens.append(self.piece_to_id(ch))
        stm = 0 if parts[1] == "w" else 1
        castling = parts[2]
        K = 1 if "K" in castling else 0
        Q = 1 if "Q" in castling else 0
        k = 1 if "k" in castling else 0
        q = 1 if "q" in castling else 0
        ep_token = 0 if parts[3] == "-" else ord(parts[3][0]) - ord("a") + 1
        halfmove_bucket = min(
            int(parts[4]) if len(parts) > 4 else 0, self.halfmove_vocab - 1
        )
        fullmove_bucket = min(
            int(parts[5]) if len(parts) > 5 else 1, self.fullmove_vocab - 1
        )

        seq = square_tokens + [
            stm,
            K,
            Q,
            k,
            q,
            ep_token,
            halfmove_bucket,
            fullmove_bucket,
        ]
        token_ids = (
            [self.square_base + v for v in seq[:64]]
            + [self.side_base + seq[64]]
            + [
                self.castle_base + seq[65],
                self.castle_base + seq[66],
                self.castle_base + seq[67],
                self.castle_base + seq[68],
            ]
            + [
                self.ep_base + seq[69],
                self.halfmove_base + seq[70],
                self.fullmove_base + seq[71],
            ]
            + [self.pad_token] * self.pad_len
        )
        return token_ids

    @staticmethod
    def piece_to_id(ch: str) -> int:
        mapping = {
            "P": 1,
            "N": 2,
            "B": 3,
            "R": 4,
            "Q": 5,
            "K": 6,
            "p": 7,
            "n": 8,
            "b": 9,
            "r": 10,
            "q": 11,
            "k": 12,
        }
        return mapping.get(ch, 0)


# -------------------------
# Latent Transformer (SOLIS Mini defaults)
# -------------------------
class LatentTransformer(nn.Module):
    def __init__(
        self,
        seq_len=77,
        d_model=512,
        vocab_size=200,
        nhead=8,
        num_layers=6,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)
        self.cls_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.LongTensor):
        b = token_ids.size(0)
        x = self.token_emb(token_ids)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb
        x = self.transformer(x)
        cls_hidden = x[:, 0, :]
        z = self.output_proj(cls_hidden)
        z = self.cls_norm(z)
        return F.normalize(z, p=2, dim=-1)


# -------------------------
# Anchored score
# -------------------------
def anchored_score_batch(
    z_children: np.ndarray, mu_black: np.ndarray, a_hat: np.ndarray
) -> np.ndarray:
    return (z_children - mu_black).dot(a_hat)


# -------------------------
# Zobrist hashing
# -------------------------
class Zobrist:
    def __init__(self, seed=2025):
        rng = np.random.RandomState(seed)
        self.piece_square = rng.randint(0, 2**63, size=(12, 64), dtype=np.int64)
        self.side = rng.randint(0, 2**63, size=(1,), dtype=np.int64)[0]
        self.castle = rng.randint(0, 2**63, size=(4,), dtype=np.int64)
        self.ep = rng.randint(0, 2**63, size=(9,), dtype=np.int64)

    def hash_board(self, board: chess.Board) -> int:
        h = np.int64(0)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                h ^= self.piece_square[piece_index(p.symbol()), sq]
        if board.turn == chess.BLACK:
            h ^= self.side
        h ^= self.castle[0] if board.has_kingside_castling_rights(chess.WHITE) else 0
        h ^= self.castle[1] if board.has_queenside_castling_rights(chess.WHITE) else 0
        h ^= self.castle[2] if board.has_kingside_castling_rights(chess.BLACK) else 0
        h ^= self.castle[3] if board.has_queenside_castling_rights(chess.BLACK) else 0
        ep = board.ep_square
        ep_idx = 0 if ep is None else chess.square_file(ep) + 1
        h ^= self.ep[ep_idx]
        return int(h & ((1 << 63) - 1))


def piece_index(symbol: str) -> int:
    mapping = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }
    return mapping[symbol]


# -------------------------
# Embedding-Guided Searcher (upgraded, SOLIS Mini compatible)
# -------------------------
class EmbeddingGuidedSearcher:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: FENTokenizer,
        mu_black: np.ndarray,
        a_hat: np.ndarray,
        device=torch.device("cpu"),
        max_depth=6,
        top_w=12,
        batch_size=128,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.mu_black = mu_black
        self.a_hat = a_hat
        self.device = device
        self.max_depth = max_depth
        self.top_w = top_w
        self.batch_size = batch_size
        self.zobrist = Zobrist()
        self.TT: Dict[int, Dict] = {}
        self.embed_cache: Dict[str, np.ndarray] = {}

    def embed_batch(self, fens: List[str]) -> np.ndarray:
        # Compute embeddings for uncached fens in batches
        uncached = [f for f in fens if f not in self.embed_cache]
        if uncached:
            tokens = [self.tokenizer.fen_to_tokens(f) for f in uncached]
            token_tensor = torch.LongTensor(tokens).to(self.device)
            with torch.no_grad():
                z = self.model(token_tensor).cpu().numpy()
            for f, vec in zip(uncached, z):
                self.embed_cache[f] = vec
        # Return embeddings in the same order as fens
        return np.array([self.embed_cache[f] for f in fens])

    def evaluate_board(self, board: chess.Board) -> float:
        z = self.embed_batch([board.fen()])[0]
        return float(
            anchored_score_batch(z[np.newaxis, :], self.mu_black, self.a_hat)[0]
        )

    def quiescence(
        self, board: chess.Board, alpha: float, beta: float, to_move: bool
    ) -> float:
        stand_pat = self.evaluate_board(board)
        if to_move:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        for move in board.legal_moves:
            if (
                board.is_capture(move)
                or board.gives_check(move)
                or board.promotion(move)
            ):
                board.push(move)
                score = self.quiescence(board, alpha, beta, not to_move)
                board.pop()
                if to_move:
                    alpha = max(alpha, score)
                else:
                    beta = min(beta, score)
                if alpha >= beta:
                    break
        return alpha if to_move else beta

    def _search_rec(
        self, board: chess.Board, depth: int, alpha: float, beta: float, to_move: bool
    ) -> float:
        if depth == 0 or board.is_game_over():
            return self.quiescence(board, alpha, beta, to_move)
        h = self.zobrist.hash_board(board)
        if h in self.TT and self.TT[h]["depth"] >= depth:
            return self.TT[h]["value"]
        moves = list(board.legal_moves)
        if to_move:
            value = -float("inf")
            ordered = self.order_moves(board, moves, True)
            for m in ordered[: self.top_w]:
                board.push(m)
                value = max(
                    value, self._search_rec(board, depth - 1, alpha, beta, False)
                )
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            ordered = self.order_moves(board, moves, False)
            for m in ordered[: self.top_w]:
                board.push(m)
                value = min(
                    value, self._search_rec(board, depth - 1, alpha, beta, True)
                )
                board.pop()
                beta = min(beta, value)
                if beta <= alpha:
                    break
        self.TT[h] = {"value": value, "depth": depth}
        return value

    def order_moves(
        self, board: chess.Board, moves: List[chess.Move], to_move: bool
    ) -> List[chess.Move]:
        # Hybrid ordering: captures first, then embedding score for non-captures
        captures = [m for m in moves if board.is_capture(m)]
        others = [m for m in moves if not board.is_capture(m)]
        fens = []
        for m in others:
            nb = board.copy(stack=False)
            nb.push(m)
            fens.append(nb.fen())
        ordered_others = []
        if fens:
            scores = self.embed_batch(fens)
            anchored = anchored_score_batch(scores, self.mu_black, self.a_hat)
            if to_move:
                idx = np.argsort(-anchored)
            else:
                idx = np.argsort(anchored)
            ordered_others = [others[i] for i in idx]
        return captures + ordered_others

    def choose_root_move(self, board: chess.Board, color: bool) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None
        best_val = -float("inf") if color else float("inf")
        best_move = None
        for m in moves:
            board.push(m)
            val = self._search_rec(board, self.max_depth - 1, -1e6, 1e6, not color)
            board.pop()
            if (color and val > best_val) or (not color and val < best_val):
                best_val = val
                best_move = m
        return best_move
