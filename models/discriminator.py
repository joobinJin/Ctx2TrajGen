import torch
import torch.nn as nn
from torch.autograd import grad

###############################################################################
# 0) Common utility: ctx_preprocess
###############################################################################
def ctx_preprocess(ctx):
    """
    Preprocess context tensor.

    Args:
        ctx (Tensor): Shape (B, T, 39) = vehicle(30) + lane(3) + exists(6)

    Steps:
        - Binarize existence flags: exists_bin = (exists_raw >= 0.5)
        - Mask vehicle features (30 = 6 neighbors * 5 feats) by exists
        - Concatenate masked vehicle, lane, and binarized exists
    """
    B, T, _ = ctx.shape
    exists_raw = ctx[..., -6:]                     # (B, T, 6)
    exists_bin = (exists_raw >= 0.5).float()       # (B, T, 6)

    veh = ctx[..., :30].view(B, T, 6, 5) * exists_bin.unsqueeze(-1)  # (B, T, 6, 5)
    veh = veh.view(B, T, 30)                                          # (B, T, 30)
    lane = ctx[..., 30:33]                                            # (B, T, 3)

    ctx_bin = torch.cat([veh, lane, exists_bin], dim=-1)              # (B, T, 39)
    return ctx_bin

###############################################################################
# 1) DiscriminatorRNN
###############################################################################
class DiscriminatorRNN(nn.Module):
    """
    WGAN-GP discriminator D(s, c, a).

    Inputs:
        s : (B, T, 6)     - state [center_x, center_y, v_x, v_y, a_x, a_y]
        c : (B, T, 39)    - context (neighbors/lane/existence flags)
        a : (B, T-1, 6)   - action (state deltas); (B, T, 6) also accepted
        seq_lens : (B,)   - actual sequence lengths

    Architecture:
        1) ctx_preprocess(c) → (B, T, 39)
        2) Embeddings: s_emb(6→128), c_emb(39→128), a_emb(6→128)
        3) Concat → (B, T-1, 384)
        4) GRU(batch_first=True, hidden_dim=128, num_layers=2)
        5) Mean-pool over time
        6) Output: (B, 1) WGAN score
    """
    def __init__(
        self,
        state_dim=6,
        context_dim=39,
        action_dim=6,
        hidden_dim=128,
        num_layers=2
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.context_dim = context_dim
        self.action_dim  = action_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        # Deeper embeddings
        self.s_emb = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.c_emb = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.a_emb = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gru = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, s, c, a, seq_lens):
        """
        Args:
            s (Tensor): (B, T, 6)
            c (Tensor): (B, T, 39)
            a (Tensor): (B, T-1, 6) or (B, T, 6)
            seq_lens (Tensor): (B,)

        Returns:
            Tensor: (B, 1) real-valued WGAN score
        """
        B, T, _ = s.shape
        T_act = a.size(1)

        if (seq_lens <= 0).any():
            raise ValueError("seq_lens contains zero or negative values.")

        # 1) Context preprocessing
        c_bin = ctx_preprocess(c)  # (B, T, 39)

        # 2) Align by action length
        if T_act < T:
            s = s[:, :T_act].contiguous()
            c_bin = c_bin[:, :T_act].contiguous()
            seq_lens = torch.clamp(seq_lens, max=T_act)
        elif T_act > T:
            raise ValueError(f"Action sequence length {T_act} exceeds state length {T}.")

        # 3) Embeddings
        s_ = self.s_emb(s.reshape(-1, self.state_dim)).reshape(B, T_act, self.hidden_dim)
        c_ = self.c_emb(c_bin.reshape(-1, self.context_dim)).reshape(B, T_act, self.hidden_dim)
        a_ = self.a_emb(a.reshape(-1, self.action_dim)).reshape(B, T_act, self.hidden_dim)

        # 4) Concatenate
        x = torch.cat([s_, c_, a_], dim=-1)  # (B, T_act, 3*hidden_dim)

        # 5) GRU with packing
        seq_lens_clamp = seq_lens.clamp(min=1)
        pack_in = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens_clamp.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(pack_in)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=T_act
        )  # (B, T_act, hidden_dim)

        # 6) Temporal mean pooling
        output_mean = torch.mean(output, dim=1)  # (B, hidden_dim)

        # 7) Output score
        out = self.fc_out(output_mean)  # (B, 1)
        return out

    def gradient_penalty(self, real_tup, fake_tup, seq_lens, device):
        """
        Compute WGAN-GP gradient penalty.

        Args:
            real_tup (tuple): (s_r, c_r, a_r)
            fake_tup (tuple): (s_f, c_f, a_f)
            seq_lens (Tensor): (B,)
            device (torch.device)
        """
        s_r, c_r, a_r = real_tup
        s_f, c_f, a_f = fake_tup

        eps = torch.rand(s_r.size(0), 1, 1, device=device)

        # Linear interpolation (lerp)
        s_mix = (eps * s_r + (1 - eps) * s_f).requires_grad_(True)
        c_mix = (eps * c_r + (1 - eps) * c_f).requires_grad_(True)
        a_mix = (eps * a_r + (1 - eps) * a_f).requires_grad_(True)

        d_mix = self.forward(s_mix, c_mix, a_mix, seq_lens)  # (B, 1)

        # Gradients w.r.t. mixed inputs
        grad_outputs = torch.ones_like(d_mix)
        grads = grad(
            outputs=d_mix,
            inputs=[s_mix, c_mix, a_mix],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        flat = torch.cat([g.reshape(g.size(0), -1) for g in grads], dim=1)
        penalty = (flat.norm(2, dim=1) - 1.0).pow(2).mean()
        return penalty

    @torch.no_grad()
    def reward(self, s, c, a, seq_lens):
        """
        GAIL-style reward: R = -D(s, c, a).
        """
        d_val = self.forward(s, c, a, seq_lens)
        return -d_val
