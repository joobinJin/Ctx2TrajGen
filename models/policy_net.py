import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def ctx_preprocess(ctx):
    """
    Context preprocessing with existence masking.

    Args:
        ctx (Tensor): (B, T, 39) = vehicle(30) + lane(3) + exists(6)

    Returns:
        Tensor: processed context of shape (B, T, 39)
    """
    ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
    B, T, _ = ctx.shape
    exists = ctx[..., -6:]                                # (B, T, 6)
    veh = ctx[..., :30].view(B, T, 6, 5) * exists.unsqueeze(-1)  # (B, T, 6, 5)
    veh = veh.view(B, T, 30)                              # (B, T, 30)
    lane = ctx[..., 30:33]                                # (B, T, 3)
    return torch.cat([veh, lane, exists], dim=-1)         # (B, T, 39)


class PolicyNetRNN(nn.Module):
    """
    RNN-based policy network with a Mixture of Gaussians head for action sampling.
    """
    def __init__(self, state_dim=6, context_dim=39, hidden_dim=128, num_layers=2, K=5, action_scale=None):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.input_dim = state_dim + context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = K

        # Use provided action_scale or a default vector
        if action_scale is not None:
            if isinstance(action_scale, torch.Tensor):
                self.action_scale = action_scale.clone().detach()
            else:
                self.action_scale = torch.tensor(action_scale)
        else:
            self.action_scale = torch.tensor([0.01, 0.01, 0.25, 0.25, 1.2, 1.2])

        self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, K * (2 * state_dim + 1))

    def forward(self, states, contexts, seq_lens, hidden=None):
        """
        Args:
            states (Tensor): (B, T, 6)
            contexts (Tensor): (B, T, 39)
            seq_lens (Tensor): (B,)
            hidden (Tensor | None): (num_layers, B, hidden_dim)

        Returns:
            mu (Tensor): (B, T, K, 6)
            log_sigma (Tensor): (B, T, K, 6)
            pi (Tensor): (B, T, K)
            hidden (Tensor): (num_layers, B, hidden_dim)

        Note:
            Masking is handled in the training loop (run_gail.py).
        """
        B, T, _ = states.shape

        # Context preprocessing
        ctx_processed = ctx_preprocess(contexts)                # (B, T, 39)
        x = torch.cat([states, ctx_processed], dim=-1)          # (B, T, 45)

        # RNN
        packed_input = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=T)  # (B, T, hidden_dim)

        # Mixture head
        out = self.fc_out(output)                               # (B, T, K * (2*6 + 1))
        out = out.view(B, T, self.K, 2 * self.state_dim + 1)    # (B, T, K, 13)

        mu = out[..., :self.state_dim]                          # (B, T, K, 6)
        log_sigma = out[..., self.state_dim:2*self.state_dim]   # (B, T, K, 6)
        log_pi = out[..., 2*self.state_dim]                     # (B, T, K)

        # Stabilize log_sigma
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=1.0)

        # Softmax with temperature on mixture weights
        pi = F.softmax(log_pi / 7.0, dim=-1)                    # (B, T, K)

        return mu, log_sigma, pi, hidden

    def sample(self, states, contexts, seq_lens, action_scale=None, hidden=None):
        """
        Sample actions from the per-timestep mixture.

        Args:
            states (Tensor): (B, T, 6)
            contexts (Tensor): (B, T, 39)
            seq_lens (Tensor): (B,)
            action_scale (Tensor | None): scale vector; if None, use self.action_scale
            hidden (Tensor | None): (num_layers, B, hidden_dim)

        Returns:
            act (Tensor): (B, T, 6)
            mu_sel (Tensor): (B, T, 6)
            log_sigma_sel (Tensor): (B, T, 6)
            comp (Tensor): (B, T)
            hidden (Tensor): (num_layers, B, hidden_dim)
        """
        B, T, _ = states.shape
        mu, log_sigma, pi, hidden = self.forward(states, contexts, seq_lens, hidden)

        # Ensure device match for action_scale
        action_scale = self.action_scale.to(states.device) if action_scale is None else action_scale.to(states.device)

        # Choose component per timestep
        comp = torch.distributions.Categorical(pi).sample()     # (B, T)
        idx = torch.arange(B, device=states.device).unsqueeze(-1).expand(-1, T)  # (B, T)

        # Select parameters
        mu_sel = mu[idx, torch.arange(T), comp]                 # (B, T, 6)
        log_sigma_sel = log_sigma[idx, torch.arange(T), comp]   # (B, T, 6)
        sigma = torch.exp(log_sigma_sel)                        # (B, T, 6)

        # Reparameterization-style sampling with clipping around mu
        eps = torch.randn_like(mu_sel)                          # (B, T, 6)
        act = mu_sel + sigma * eps                              # (B, T, 6)
        act = torch.clamp(act, min=mu_sel - 5 * sigma, max=mu_sel + 5 * sigma)

        # Squash to [-1, 1] and scale per-dimension
        act = torch.tanh(act) * action_scale                    # (B, T, 6)

        return act, mu_sel, log_sigma_sel, comp, hidden


class ValueNetRNN(nn.Module):
    """
    RNN-based value network for PPO/GAE.
    """
    def __init__(self, state_dim=6, context_dim=39, hidden_dim=128, num_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.input_dim = state_dim + context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, states, contexts, seq_lens, hidden=None):
        """
        Args:
            states (Tensor): (B, T, 6)
            contexts (Tensor): (B, T, 39)
            seq_lens (Tensor): (B,)
            hidden (Tensor | None): (num_layers, B, hidden_dim)

        Returns:
            val (Tensor): (B, T, 1)
            hidden (Tensor): (num_layers, B, hidden_dim)

        Note:
            Masking is handled in the training loop.
        """
        B, T, _ = states.shape

        ctx_processed = ctx_preprocess(contexts)                 # (B, T, 39)
        x = torch.cat([states, ctx_processed], dim=-1)          # (B, T, 45)

        packed_input = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=T)  # (B, T, hidden_dim)

        val = self.fc_value(output)                              # (B, T, 1)
        return val, hidden
