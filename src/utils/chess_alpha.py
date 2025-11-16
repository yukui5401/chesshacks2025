import pytorch_lightning as pl
import chess
import numpy as np
import torch


# ------------------------
# FEN -> tensor
# ------------------------
def fen_to_tensor(fen):
    board = chess.Board(fen)
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }[piece.piece_type]
        if piece.color == chess.BLACK:
            plane_idx += 6
        planes[plane_idx, row, col] = 1
    return planes


# ------------------------
# Residual Block
# ------------------------
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# ------------------------
# AlphaZero-style ResNet
# ------------------------
class ChessAlphaZeroResNet(pl.LightningModule):
    def __init__(
        self,
        board_planes=12,
        hidden_channels=128,
        num_res_blocks=5,
        num_moves=4672,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conv_in = torch.nn.Sequential(
            torch.nn.Conv2d(board_planes, hidden_channels, 3, padding=1),
            torch.nn.BatchNorm2d(hidden_channels),
            torch.nn.ReLU(),
        )

        self.res_blocks = torch.nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, 2, 1),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 8 * 8, num_moves),
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, 1, 1),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh(),
        )

        self.policy_loss_fn = torch.nn.CrossEntropyLoss()
        self.value_loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def training_step(self, batch, batch_idx):
        boards, policy_targets, value_targets = batch
        pred_policy, pred_value = self(boards)
        policy_loss = self.policy_loss_fn(pred_policy, policy_targets)
        value_loss = self.value_loss_fn(pred_value, value_targets)
        loss = policy_loss + value_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        boards, policy_targets, value_targets = batch
        pred_policy, pred_value = self(boards)
        policy_loss = self.policy_loss_fn(pred_policy, policy_targets)
        value_loss = self.value_loss_fn(pred_value, value_targets)
        loss = policy_loss + value_loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
