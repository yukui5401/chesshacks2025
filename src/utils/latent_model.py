import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# FEN → tensor (pieces + extras)
# -------------------------
def fen_to_tensor(fen):
    """
    Convert FEN into tensor of shape [14, 8, 8]:
    - 12 planes: pieces (P,N,B,R,Q,K,p,n,b,r,q,k)
    - 1 plane: active color (1 for white, 0 for black)
    - 1 plane: castling rights (KQkq)
    """
    PIECE_TO_IDX = {
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
    planes = torch.zeros(14, 8, 8)

    board_part, active_color, castling, ep, *_ = fen.split()
    rows = board_part.split("/")
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                planes[PIECE_TO_IDX[char], r, c] = 1
                c += 1

    # Active color plane
    planes[12, :, :] = 1 if active_color == "w" else 0

    # Castling rights plane: encode KQkq as 4 bits over plane
    castling_map = {"K": 0, "Q": 1, "k": 2, "q": 3}
    for ch in castling:
        if ch in castling_map:
            planes[13, :, :] += 1 << castling_map[ch]

    return planes  # shape [14,8,8]


# -------------------------
# Latent Encoder
# -------------------------
class LatentEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Conv layers over 14x8x8 planes
        self.conv = nn.Sequential(
            nn.Conv2d(14, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        """
        x: [batch, 14, 8, 8]
        returns: L2-normalized latent embedding [batch, latent_dim]
        """
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        return F.normalize(z, p=2, dim=-1)

    def supervised_contrastive_loss(z, labels, tau=0.1):
        device = z.device
        batch_size = z.shape[0]
        sim_matrix = torch.matmul(z, z.T) / tau
        exp_sim = torch.exp(sim_matrix)
        loss = 0.0
        for i in range(batch_size):
            mask_pos = (labels == labels[i]) & (
                torch.arange(batch_size, device=device) != i
            )
            mask_all = torch.ones(batch_size, dtype=torch.bool, device=device)
            mask_all[i] = False
            numerator = exp_sim[i][mask_pos].sum()
            denominator = exp_sim[i][mask_all].sum()
            if numerator > 0:
                loss += -torch.log(numerator / denominator)
        return loss / batch_size
