import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchdiffeq import odeint 
from torch_geometric.utils import to_dense_adj 


def positional_encoding_1d(time, embedding_dim, max_positions=5000):
    """
    Standard sinusoidal positional encoding for a single time scalar in [0, 1].
    Scales `time` by max_positions to mimic discrete index embeddings.
    Returns a [embedding_dim] vector.
    """
    # scale time
    scaled_time = time * max_positions  
    half_dim = embedding_dim // 2
    # define frequency exponents
    div_term = math.log(max_positions) / (half_dim - 1) if half_dim > 1 else 1.0
    freq_factors = torch.exp(
        -div_term * torch.arange(half_dim, dtype=torch.float32, device=time.device)
    )  # [half_dim]
    # elementwise multiply & build [sin, cos]
    angles = scaled_time * freq_factors  # [half_dim]
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    emb = torch.cat([sin_part, cos_part], dim=-1)  # [embedding_dim <= 2*half_dim]
    # if embedding_dim is odd, zero-pad
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb

def rope_encoding_1d(time, embedding_dim, max_positions=5000):
    """
    A more typical RoPE implementation for a single time scalar in [0, 1].
    This version *generates* a 1D embedding by essentially rotating a base sinusoidal 
    representation within each pair of dimensions.

    In full transformer usage, RoPE is typically applied by *rotating an existing embedding* 
    rather than generating one from scratch. For demonstration, here we:
      1. Build the standard sinusoidal encoding: angles = time * freq
      2. Split those angles into pairs
      3. For each pair, apply a rotation that yields a (cos θ, sin θ) style embedding.

    Returns:
        Tensor of shape [embedding_dim].
    """
    # 1) Generate the standard sinusoidal angles
    scaled_time = time * max_positions
    half_dim = embedding_dim // 2
    if half_dim < 1:
        raise ValueError(f"embedding_dim must be >= 2 for RoPE, got {embedding_dim}")
    div_term = math.log(max_positions) / (half_dim - 1) if half_dim > 1 else 1.0
    
    # frequencies
    freq_factors = torch.exp(
        -div_term * torch.arange(half_dim, dtype=torch.float32, device=time.device)
    )  # [half_dim]
    angles = scaled_time * freq_factors  # [half_dim], each entry is θ_i
    
    # 2) For each i, create a 2D rotation of a "base" vector. 
    #    Usually in practice, you'd rotate an existing pair [x_2i, x_2i+1] by θ_i.
    #    Here, we just produce [cos(θ_i), sin(θ_i)] for demonstration.
    cos_theta = torch.cos(angles)  # [half_dim]
    sin_theta = torch.sin(angles)  # [half_dim]

    # 3) Interleave cos and sin to get final embedding
    rope_emb = torch.zeros(embedding_dim, device=time.device)
    rope_emb[0::2] = cos_theta
    rope_emb[1::2] = sin_theta

    # If embedding_dim is odd, pad last dimension (similar to the positional_encoding_1d approach)
    if embedding_dim % 2 == 1:
        rope_emb = F.pad(rope_emb, (0, 1), mode='constant')

    return rope_emb

class GraphFourier(nn.Module): 
    def __init__(self, U_batch, B, N, T): 
        """
        Computes the graph Fourier basis (eigenvectors of the Laplacian) for each graph in the batch.

        Args:
            U_batch: eigenvectors of the Laplacian for each graph in the batch, shape: [B, N, N].
            B: Batch size. 
            N: int, number of nodes per graph.

        Returns:
            U_batch: Tensor of shape [B, N, N], the graph Fourier basis for each graph.
        """
        super(GraphFourier, self).__init__()
        self.B = B 
        self.N = N
        self.T = T 

        self.U_batch = U_batch  # [B, N, N]

    def transform(self, x): 
        """
        Performs the graph Fourier transform on batched graph signals.

        Args:
            x: Tensor of shape [B, N, E], graph signals for each graph in the batch.

        Returns:
            x_hat: Tensor of shape [B, N, E], graph Fourier coefficients for each graph.
        """
        # x: [B, N, E]
        # U_batch: [B, N, N]

        # Transpose U_batch: [B, N, N]
        U_transpose = self.U_batch.transpose(1, 2)  # [B, N, N]

        # Perform batched matrix multiplication
        x_hat = torch.einsum('bni,bie->bne', U_transpose, x)  # [B, N, E]

        return x_hat
    
    def inverse_transform(self, x_hat): 
        """
        Performs the inverse graph Fourier transform on batched graph signals.

        Args:
            x_hat: Tensor of shape [B, T, M, E], graph Fourier coefficients for each graph.

        Returns:
            x: Tensor of shape [B, T, N, E], reconstructed graph signals for each graph.
        """
        # x_hat: [B, T, M, E]
        # U_batch: [B, N, N]

        M = x_hat.size(2)  # Number of frequency modes 
        assert self.T == x_hat.size(1), "Number of time points should be equal to T" 
        assert M <= self.N, "Number of frequency modes should be less than or equal to N" 
        if M < self.N: 
            zeros = torch.zeros(self.B, self.T, self.N - M, x_hat.size(-1), device=x_hat.device)
            x_hat = torch.cat([x_hat, zeros], dim=2)

        U_batch_expanded = self.U_batch.unsqueeze(1).expand(-1, self.T, -1, -1)  # [B, T, N, N]
        U_batch_expanded = U_batch_expanded.reshape(self.B * self.T, self.N, self.N)  # [BT, N, N] 
        x_hat = x_hat.reshape(self.B * self.T, self.N, -1)  # [BT, N, E] 

        # Perform batched matrix multiplication
        x = torch.einsum('bni,bie->bne', U_batch_expanded, x_hat) # [BT, N, E]

        x = x.reshape(self.B, self.T, self.N, -1)  # [B, T, N, E] 

        return x



# Updated get_timestep_embedding function
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: Tensor of arbitrary shape containing timestep indices.
        embedding_dim: The dimensionality of the embeddings.
        max_positions: The maximum number of positions (default=10000).
    Returns:
        A tensor of shape [*timesteps.shape, embedding_dim] containing the embeddings.
    """
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[..., None] * emb  # Shape: [*timesteps.shape, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # Shape: [*timesteps.shape, embedding_dim]
    if embedding_dim % 2 == 1:  # Zero pad if embedding_dim is odd
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb

# TODO: nfe ?????? 

@torch.jit.script
def compl_mul1d(a, b):    
    # (M, B, in_ch), (in_ch, out_ch, M) -> (M, B, out_ch)
    return torch.einsum("mni,iom->mno", a, b)

class ODEFunction_complex(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(ODEFunction_complex, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = (1 / (in_ch * out_ch))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float)
        )

    def forward(self, t, x_ft):
        assert x_ft.shape[0] == self.modes1 # Check if the number of modes is correct
        # Neural ODE function similar to compl_mul1d but with learned weights
        out_ft = compl_mul1d(x_ft, torch.view_as_complex(self.weights1))
        return out_ft

class ODEFunction_real(nn.Module):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        modes1, 
        mode_interaction='no_interaction', 
        n_heads=1, 
        hidden_dim=128,
        # ---- NEW ARGS ----
        time_mode='none', 
        time_embedding_dim=16, 
        max_positions=5000
    ):
        """
        ODEFunction_real now has optional mode_interaction and optional time_mode:
            mode_interaction in ['no_interaction', 'attention', 'concat']
            time_mode in ['none', 'concat', 'mlp', 'posenc', 'rope']

        Args:
            in_ch: int, input channel dimension
            out_ch: int, output channel dimension
            modes1: int, number of modes
            mode_interaction: str, how to handle mode interaction
            n_heads: int, number of attention heads if using attention
            hidden_dim: int, hidden dimension for MLP if using 'concat'
            time_mode: str, how to incorporate time
            time_embedding_dim: int, dimension of time embedding if using 'concat', 'mlp', or encodings
            max_positions: int, used for positional/rope embeddings
        """
        super(ODEFunction_real, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.scale = (1 / (in_ch * out_ch))

        self.mode_interaction = mode_interaction
        self.time_mode = time_mode
        self.time_embedding_dim = time_embedding_dim
        self.max_positions = max_positions

        print(f"time_mode in ODEFunction_real: {self.time_mode}")
        print(f"mode_interaction in ODEFunction_real: {self.mode_interaction}")

        # Adjust effective input channels if we plan to concat time
        effective_in_ch = in_ch
        if self.time_mode != 'none':
            effective_in_ch += time_embedding_dim

        # Original weights for direct mode-wise multiplication
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.modes1, effective_in_ch, out_ch, dtype=torch.float)
        )

        # -- Self-Attention if requested --
        if self.mode_interaction == 'attention':
            self.attn = nn.MultiheadAttention(embed_dim=effective_in_ch, num_heads=n_heads, batch_first=False)
            self.post_attn_linear = nn.Linear(effective_in_ch, effective_in_ch)

        # -- Concatenate + MLP if requested --
        elif self.mode_interaction == 'concat':
            self.mlp = nn.Sequential(
                nn.Linear(self.modes1 * effective_in_ch, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.modes1 * effective_in_ch),
            )

        # MLP for time if time_mode == 'mlp'
        if self.time_mode == 'mlp':
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, time_embedding_dim),
            )

    def _build_time_embedding(self, t, B):
        """
        Build a time embedding of dimension self.time_embedding_dim
        for a scalar t in [0,1], then replicate for batch B.
        Return shape: [B, time_embedding_dim].
        """
        # t is typically a scalar float from odeint
        # We'll unsqueeze(0) to create a shape [1], convert to a tensor if needed
        if not torch.is_tensor(t):
            t_val = torch.tensor([t], dtype=torch.float, device=self.weights1.device)
        else:
            t_val = t if t.dim() > 0 else t.unsqueeze(0)

        if self.time_mode == 'none':
            # no time embedding
            return None

        elif self.time_mode == 'concat':
            # Just replicate t as a scalar, or create a [1]-dim MLP?
            # Here we'll do direct: shape [B, 1]
            # Then tile to shape [B, time_embedding_dim]
            # (Though typically you'd want an MLP or embedding.)
            t_embed = t_val.repeat(B, 1)  # [B, 1]
            # expand to dimension time_embedding_dim
            t_embed = t_embed.repeat(1, self.time_embedding_dim)  # [B, time_embedding_dim]
            return t_embed

        elif self.time_mode == 'mlp':
            # time_mlp expects shape [B, 1]
            t_input = t_val.repeat(B).unsqueeze(1)  # [B, 1]
            t_embed = self.time_mlp(t_input)        # [B, time_embedding_dim]
            return t_embed

        elif self.time_mode == 'posenc':
            # build a single [time_embedding_dim] from positional_encoding_1d
            # then replicate across batch
            pe = positional_encoding_1d(t_val, self.time_embedding_dim, max_positions=self.max_positions)
            t_embed = pe.repeat(B, 1)  # [B, time_embedding_dim]
            return t_embed

        elif self.time_mode == 'rope':
            # build a single [time_embedding_dim] from rope_encoding_1d
            # then replicate across batch
            re = rope_encoding_1d(t_val, self.time_embedding_dim, max_positions=self.max_positions)
            t_embed = re.repeat(B, 1)  # [B, time_embedding_dim]
            return t_embed

    def forward(self, t, x_hat):
        """
        x_hat: [M, B, in_ch]
        Returns: [M, B, out_ch]
        """
        # Step 0: Possibly incorporate time
        M, B, C = x_hat.shape

        if self.time_mode != 'none':
            # build time embedding
            t_embed = self._build_time_embedding(t, B)  # [B, time_embedding_dim] or None
            if t_embed is not None:
                # expand to [M, B, time_embedding_dim]
                t_embed_expanded = t_embed.unsqueeze(0).expand(M, -1, -1)
                # concat along the last dimension
                x_hat = torch.cat([x_hat, t_embed_expanded], dim=-1)  # [M, B, in_ch + time_embedding_dim]

        # Step 1: Manage interaction among modes
        if self.mode_interaction == 'attention':
            attn_out, _ = self.attn(x_hat, x_hat, x_hat)  # [M, B, effective_in_ch]
            x_hat = self.post_attn_linear(attn_out)       # [M, B, effective_in_ch]

        elif self.mode_interaction == 'concat':
            # Flatten the modes dimension for each batch
            x_hat_perm = x_hat.permute(1, 0, 2)  # [B, M, effective_in_ch]
            B_, M_, C_ = x_hat_perm.shape
            flattened = x_hat_perm.reshape(B_, M_ * C_)  # [B, M*effective_in_ch]
            updated = self.mlp(flattened)                # [B, M*effective_in_ch]
            updated = updated.reshape(B_, M_, C_)        # [B, M, effective_in_ch]
            x_hat = updated.permute(1, 0, 2)             # [M, B, effective_in_ch]

        # Step 2: Original mode-wise linear transform
        out_hat = torch.einsum("mni,mio->mno", x_hat, self.weights1)  # [M, B, out_ch]
        return out_hat

class ODEFunction_real_x(nn.Module):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        modes1, 
        mode_interaction='no_interaction', 
        n_heads=1, 
        hidden_dim=128,
        # ---- NEW ARGS ----
        time_mode='none',
        time_embedding_dim=16,
        max_positions=5000
    ):
        """
        ODEFunction_real_x now supports optional mode_interaction and time_mode for
        input shape [M, B, E, in_ch].

        time_mode in ['none', 'concat', 'mlp', 'posenc', 'rope']
        """
        super(ODEFunction_real_x, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.scale = (1 / (in_ch * out_ch))

        self.mode_interaction = mode_interaction
        self.time_mode = time_mode
        self.time_embedding_dim = time_embedding_dim
        self.max_positions = max_positions

        print(f"time_mode in ODEFunction_real_x: {self.time_mode}")
        print(f"mode_interaction in ODEFunction_real_x: {self.mode_interaction}")

        # Adjust effective input channels if we plan to concat time
        effective_in_ch = in_ch
        if self.time_mode != 'none':
            effective_in_ch += time_embedding_dim

        # Weights shape: [modes1, effective_in_ch, out_ch]
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.modes1, effective_in_ch, self.out_ch, dtype=torch.float)
        )

        if self.mode_interaction == 'attention':
            self.attn = nn.MultiheadAttention(embed_dim=effective_in_ch, num_heads=n_heads, batch_first=False)
            self.post_attn_linear = nn.Linear(effective_in_ch, effective_in_ch)

        elif self.mode_interaction == 'concat':
            self.mlp = nn.Sequential(
                nn.Linear(self.modes1 * effective_in_ch, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.modes1 * effective_in_ch),
            )

        if self.time_mode == 'mlp':
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, time_embedding_dim),
            )

    def _build_time_embedding(self, t, B):
        """
        Create time embedding (similar to ODEFunction_real).
        Return shape: [B, time_embedding_dim] or None if time_mode='none'.
        """
        if not torch.is_tensor(t):
            t_val = torch.tensor([t], dtype=torch.float, device=self.weights1.device)
        else:
            t_val = t if t.dim() > 0 else t.unsqueeze(0)

        if self.time_mode == 'none':
            return None
        elif self.time_mode == 'concat':
            # shape [B, time_embedding_dim]
            t_embed = t_val.repeat(B, 1)
            t_embed = t_embed.repeat(1, self.time_embedding_dim)
            return t_embed
        elif self.time_mode == 'mlp':
            t_input = t_val.repeat(B).unsqueeze(1)  # [B, 1]
            t_embed = self.time_mlp(t_input)        # [B, time_embedding_dim]
            return t_embed
        elif self.time_mode == 'posenc':
            pe = positional_encoding_1d(t_val, self.time_embedding_dim, max_positions=self.max_positions)
            t_embed = pe.repeat(B, 1)
            return t_embed
        elif self.time_mode == 'rope':
            re = rope_encoding_1d(t_val, self.time_embedding_dim, max_positions=self.max_positions)
            t_embed = re.repeat(B, 1)
            return t_embed

    def forward(self, t, x_hat):
        """
        x_hat: [M, B, E, in_ch]
        Returns: [M, B, E, out_ch]
        """
        M, B, E, C = x_hat.shape

        # Step 0: Possibly incorporate time
        if self.time_mode != 'none':
            t_embed = self._build_time_embedding(t, B)  # [B, time_embedding_dim] or None
            if t_embed is not None:
                # We want to concat along last dimension => shape [M, B, E, in_ch + time_embedding_dim]
                # First, replicate t_embed across M, E
                # t_embed: [B, time_embedding_dim]
                # expand to [M, B, E, time_embedding_dim]
                t_embed_expanded = t_embed.unsqueeze(0).unsqueeze(2).expand(M, -1, E, -1)
                x_hat = torch.cat([x_hat, t_embed_expanded], dim=-1)

        # Step 1: Manage interaction among modes
        if self.mode_interaction == 'attention':
            # Flatten out B,E => shape [M, B*E, effective_in_ch]
            M_, BE_, C_ = M, B * E, x_hat.shape[-1]
            x_flat = x_hat.reshape(M_, BE_, C_)
            attn_out, _ = self.attn(x_flat, x_flat, x_flat)  # [M, B*E, effective_in_ch]
            x_flat = self.post_attn_linear(attn_out)
            x_hat = x_flat.reshape(M, B, E, C_)

        elif self.mode_interaction == 'concat':
            # [M, B, E, effective_in_ch] -> [B, E, M, effective_in_ch]
            x_perm = x_hat.permute(1, 2, 0, 3)  
            B_, E_, M_, C_ = x_perm.shape
            flattened = x_perm.reshape(B_ * E_, M_ * C_)     # [B*E, M*effective_in_ch]
            updated = self.mlp(flattened)                    # [B*E, M*effective_in_ch]
            updated = updated.reshape(B_, E_, M_, C_)        # [B, E, M, effective_in_ch]
            x_hat = updated.permute(2, 0, 1, 3)              # [M, B, E, effective_in_ch]

        # Step 2: Original mode-wise linear transform
        # weights1: [M, effective_in_ch, out_ch]
        out_hat = torch.einsum('mbei,mio->mbeo', x_hat, self.weights1)
        return out_hat

class SpectralConv1dODE(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, solver, rtol, atol, fourier_basis=None, 
                 no_fourier=False, no_ode=False, mode_interaction="no_interaction",
                 time_mode='none',
                 time_embedding_dim=16,
                 max_positions=5000):


        super(SpectralConv1dODE, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.solver = solver 
        self.rtol = rtol 
        self.atol = atol 
        self.fourier_basis = fourier_basis 

        self.no_fourier = no_fourier
        self.no_ode = no_ode 
        self.modes1 = modes1 
        self.mode_interaction = mode_interaction 

        self.time_mode = time_mode
        self.time_embedding_dim = time_embedding_dim
        self.max_positions = max_positions

        assert not (no_fourier and no_ode), "no_fourier and no_ode cannot be True at the same time" 

        
        if no_fourier is True:
            self.ode_func = ODEFunction_real(in_ch, out_ch, modes1, mode_interaction=self.mode_interaction, time_mode=self.time_mode, time_embedding_dim=self.time_embedding_dim, max_positions=self.max_positions) 
        elif fourier_basis == "linear": 
            self.ode_func = ODEFunction_complex(in_ch, out_ch, modes1)
        elif fourier_basis == "graph":
            self.ode_func = ODEFunction_real(in_ch, out_ch, modes1, mode_interaction=self.mode_interaction, time_mode=self.time_mode, time_embedding_dim=self.time_embedding_dim, max_positions=self.max_positions) 
        else: 
            raise ValueError("fourier_basis must be either 'linear' or 'graph'")
        

    def forward(self, x, times, U_batch=None):
        """
        Args:
            x: Tensor of shape [B, N, E], where B is batch size, N is the spatial dimension, E is the number of features.
            times: Tensor of shape [B, T], containing time points for each batch element.
        Returns:
            x_pred: Tensor of shape [T, B, N, E], containing the predicted values at the specified time points.
        """
        # x: [B, N, E]
        # times: [B, T]

        x = x.swapaxes(0, 1)  # Swap batch and particle dimensions for Fourier transform, now x is [N, B, E]

        N, B, E = x.shape  # Spatial dimension N, batch size B, feature dimension E

        T = times.shape[1]  # Number of time points 

        if self.no_fourier: 
            assert N == self.modes1, "Number of modes should be equal to N" 

        if self.fourier_basis == "graph":
            gft = GraphFourier(U_batch, B, N, T)

        # Compute Fourier coefficients along the spatial dimension
        with torch.cuda.amp.autocast(enabled=False):
            if self.no_fourier is True: 
                x_ft = x  # [N, B, E] 
            elif self.fourier_basis == "linear": 
                x_ft = torch.fft.rfftn(x.float(), dim=[0])  # FFT along the N dimension, x_ft shape: [N_freq, B, E]
            elif self.fourier_basis == "graph": 
                x_ft = gft.transform(x.swapaxes(0, 1)).swapaxes(0, 1) # [N, B, E]
            x_ft = x_ft[:self.modes1]  # Keep only the first 'modes1' frequency modes, x_ft shape: [M, B, E]
            M = x_ft.shape[0]  # Number of modes retained

            if self.no_ode is True: 
                x_ft_pred = self.ode_func(t=None, x_hat=x_ft) # [N, B, E] 
                # repeat the output for T times, and switch to shape [B, T, N, E]
                x_ft_pred_desired = x_ft_pred.permute(1, 0, 2).unsqueeze(1).repeat(1, T, 1, 1) # [B, T, N, E] 
            else:  
                # Unionize all the time points across the batch
                t_union, inverse_indices = torch.unique(times.flatten(), sorted=True, return_inverse=True)
                indices = inverse_indices.view(B, -1)  # Reshape to [B, T], mapping each time to its index in t_union

                # make sure t_union starts with 0 
                starts_with_zero = (t_union[0] == 0)
                assert not starts_with_zero, "t_union should not start with 0" 
                if not starts_with_zero:
                    t_union = torch.cat((torch.tensor([0.0], device=x.device), t_union)) # Add 0.0 to the beginning of t_union 

                # Solve the ODE in Fourier space over the unionized times
                x_ft_pred = odeint(
                    self.ode_func,
                    x_ft,  # Initial condition: [M, B, E]
                    t_union,  # Unioned time points: [L], where L is the number of unique time points
                    method=self.solver,
                    rtol=self.rtol,
                    atol=self.atol
                )  # Output shape: [L, M, B, E]

                if not starts_with_zero:
                    x_ft_pred = x_ft_pred[1:] 

                # Rearrange x_ft_pred for indexing
                x_ft_pred = x_ft_pred.permute(2, 0, 1, 3)  # Now x_ft_pred is [B, L, M, E]

                # Prepare indices for gathering the desired time points
                B, L, M, E = x_ft_pred.shape  # Extract dimensions
                x_ft_pred_flat = x_ft_pred.reshape(B, L, M * E)  # Flatten the last two dimensions: [B, L, M*E]

                indices_expanded = indices.unsqueeze(-1)  # Expand indices to shape [B, T, 1]

                # Gather the outputs corresponding to the desired time points for each batch element
                x_ft_pred_desired_flat = x_ft_pred_flat.gather(1, indices_expanded.expand(-1, -1, M * E))  # [B, T, M*E]

                # Reshape back to original dimensions
                x_ft_pred_desired = x_ft_pred_desired_flat.reshape(B, -1, M, E)  # [B, T, M, E]



            # Inverse FFT to return to physical space
            if self.no_fourier is True: 
                x_pred = x_ft_pred_desired.permute(1, 0, 2, 3)  # [T, B, N, E]
            elif self.fourier_basis == "linear": 
                # Permute to match expected shape for inverse FFT
                x_ft_pred_desired = x_ft_pred_desired.permute(1, 2, 0, 3)  # [T, M, B, E]
                x_pred = torch.fft.irfftn(x_ft_pred_desired, s=[N], dim=[1])  # Inverse FFT along the modes dimension, [T, N, B, E]
                x_pred = x_pred.permute(0, 2, 1, 3)  # Rearrange to [T, B, N, E]
            elif self.fourier_basis == "graph": 
                x_pred = gft.inverse_transform(x_ft_pred_desired).swapaxes(0, 1) # [T, B, N, E] 
            

        return x_pred

class TimeConvODE(nn.Module):
    def __init__(self, in_ch, out_ch, modes, act, solver, rtol, atol, fourier_basis=None, 
                 no_fourier=False, no_ode=False, mode_interaction="no_interaction",
                 time_mode='none',
                 time_embedding_dim=16,
                 max_positions=5000):
        super(TimeConvODE, self).__init__()
        self.t_conv = SpectralConv1dODE(in_ch, out_ch, modes, solver, rtol, atol, fourier_basis, 
                                        no_fourier=no_fourier, no_ode=no_ode, mode_interaction=mode_interaction, 
                                        time_mode=time_mode, time_embedding_dim=time_embedding_dim, max_positions=max_positions)
        self.act = act

        self.no_fourier = no_fourier
        self.no_ode = no_ode
        self.time_mode = time_mode
        self.time_embedding_dim = time_embedding_dim
        self.max_positions = max_positions

    def forward(self, x, times, U_batch=None):
        h = self.t_conv(x, times, U_batch)
        out = self.act(h)
        # pls use broadcasting for the following line to work 
        # add one more dimension to the tensor to make it [B, N, T, E] 
        x = x.unsqueeze(0) # [B, N, 1, E] 
        return x + out
    

    

@torch.jit.script
def compl_mul1d_x(a, b):
    # (M, B, E, in_ch) x (in_ch, out_ch, M) -> (M, B, E, out_ch)
    return torch.einsum("mndi,iom->mndo", a, b)

class ODEFunction_complex_x(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(ODEFunction_complex_x, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = (1 / (in_ch * out_ch))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float)
        )

    def forward(self, t, x_ft):
        assert x_ft.shape[0] == self.modes1 # Check if the number of modes is correct
        # Neural ODE function similar to compl_mul1d but with learned weights
        out_ft = compl_mul1d_x(x_ft, torch.view_as_complex(self.weights1))
        return out_ft


class SpectralConv1dODE_x(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, solver, rtol, atol, fourier_basis=None, 
                 no_fourier=False, no_ode=False, mode_interaction="no_interaction",
                 time_mode='none',
                 time_embedding_dim=16,
                 max_positions=5000):
        super(SpectralConv1dODE_x, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.fourier_basis = fourier_basis 

        self.no_fourier = no_fourier
        self.no_ode = no_ode
        self.modes1 = modes1 
        self.mode_interaction = mode_interaction 
        self.time_mode = time_mode
        self.time_embedding_dim = time_embedding_dim
        self.max_positions = max_positions

        assert not (no_fourier and no_ode), "no_fourier and no_ode cannot be True at the same time" 

        if no_fourier is True:
            self.ode_func = ODEFunction_real_x(in_ch, out_ch, modes1, mode_interaction=self.mode_interaction, time_mode=self.time_mode, time_embedding_dim=self.time_embedding_dim, max_positions=self.max_positions) 
        elif fourier_basis == "linear": 
            self.ode_func = ODEFunction_complex_x(in_ch, out_ch, modes1)
        elif fourier_basis == "graph":
            self.ode_func = ODEFunction_real_x(in_ch, out_ch, modes1, mode_interaction=self.mode_interaction, time_mode=self.time_mode, time_embedding_dim=self.time_embedding_dim, max_positions=self.max_positions)
        else:
            raise ValueError("fourier_basis must be either 'linear' or 'graph'") 

    def forward(self, x, times, U_batch=None):
        """
        Args:
            x: Tensor of shape [B, N, E], where B is batch size, N is the spatial dimension, E is the number of features.
            times: Tensor of shape [B, T], containing time points for each batch element.
        Returns:
            x_pred: Tensor of shape [T, B, N, E], containing the predicted values at the specified time points.
        """
        # x: [B, N, E]
        # times: [B, T]

        x = x.swapaxes(0, 1)  # Swap batch and particle dimensions for Fourier transform, now x is [N, B, E]

        N, B, E, _ = x.shape  # Spatial dimension N, batch size B, feature dimension E, in_ch=2 

        T = times.shape[1]  # Number of time points 

        if self.no_fourier: 
            assert N == self.modes1, "Number of modes should be equal to N" 

        if self.fourier_basis == "graph":
            gft = GraphFourier(U_batch, B, N, T)

        # Compute Fourier coefficients along the spatial dimension
        with torch.cuda.amp.autocast(enabled=False):
            if self.no_fourier is True: 
                x_ft = x  # [N, B, E, in_ch=2] 
            elif self.fourier_basis == "linear": 
                x_ft = torch.fft.rfftn(x.float(), dim=[0])  # FFT along the N dimension, x_ft shape: [N_freq, B, E, in_ch=2]
            elif self.fourier_basis == "graph": 
                x_ft = gft.transform(x.swapaxes(0, 1).reshape(B, N, -1)).swapaxes(0, 1).reshape(N, B, E, -1) # [N, B, E, in_ch=2]

            x_ft = x_ft[:self.modes1]  # Keep only the first 'modes1' frequency modes, x_ft shape: [M, B, E, in_ch=2]
            M = x_ft.shape[0]  # Number of modes retained

            if self.no_ode is True: 
                x_ft_pred = self.ode_func(t=None, x_hat=x_ft) # [N, B, E, out_ch=2] 
                # repeat the output for T times, and switch to shape [B, T, N, E]
                x_ft_pred_desired = x_ft_pred.permute(1, 0, 2, 3).unsqueeze(1).repeat(1, T, 1, 1, 1) # [B, T, N, E, out_ch=2] 
            else:
                # Unionize all the time points across the batch
                t_union, inverse_indices = torch.unique(times.flatten(), sorted=True, return_inverse=True)
                indices = inverse_indices.view(B, -1)  # Reshape to [B, T], mapping each time to its index in t_union

                # make sure t_union starts with 0 
                starts_with_zero = (t_union[0] == 0)
                assert not starts_with_zero, "t_union should not start with 0" 
                if not starts_with_zero:
                    t_union = torch.cat((torch.tensor([0.0], device=x.device), t_union)) # Add 0.0 to the beginning of t_union 

                # Solve the ODE in Fourier space over the unionized times
                x_ft_pred = odeint(
                    self.ode_func,
                    x_ft,  # Initial condition: [M, B, E, in_ch=2]
                    t_union,  # Unioned time points: [L], where L is the number of unique time points
                    method=self.solver,
                    rtol=self.rtol,
                    atol=self.atol
                )  # Output shape: [L, M, B, E, out_ch=2]

                if not starts_with_zero:
                    x_ft_pred = x_ft_pred[1:] 

                # Rearrange x_ft_pred for indexing
                x_ft_pred = x_ft_pred.permute(2, 0, 1, 3, 4)  # Now x_ft_pred is [B, L, M, E, out_ch=2]

                # Prepare indices for gathering the desired time points
                B, L, M, E, _ = x_ft_pred.shape  # Extract dimensions
                x_ft_pred_flat = x_ft_pred.reshape(B, L, M * E * self.out_ch)  # Flatten the last two dimensions: [B, L, M*E]

                indices_expanded = indices.unsqueeze(-1)  # Expand indices to shape [B, T, 1]

                # Gather the outputs corresponding to the desired time points for each batch element
                x_ft_pred_desired_flat = x_ft_pred_flat.gather(1, indices_expanded.expand(-1, -1, M * E * self.out_ch))  # [B, T, M*E*out_ch]

                # Reshape back to original dimensions
                x_ft_pred_desired = x_ft_pred_desired_flat.reshape(B, -1, M, E, self.out_ch)  # [B, T, M, E, out_ch]

            if self.no_fourier is True: 
                x_pred = x_ft_pred_desired.permute(1, 0, 2, 3, 4)  # [T, B, N, E, out_ch]
            elif self.fourier_basis == "linear": 
                # Permute to match expected shape for inverse FFT
                x_ft_pred_desired = x_ft_pred_desired.permute(1, 2, 0, 3, 4)  # [T, M, B, E, out_ch]

                # Inverse FFT to return to physical space
                x_pred = torch.fft.irfftn(x_ft_pred_desired, s=[N], dim=[1])  # Inverse FFT along the modes dimension, [T, N, B, E, out_ch]
                x_pred = x_pred.permute(0, 2, 1, 3, 4)  # Rearrange to [T, B, N, E, out_ch]
            elif self.fourier_basis == "graph": 
                x_pred = gft.inverse_transform(x_ft_pred_desired.view(B, T, M, -1)).swapaxes(0, 1).reshape(T, B, N, E, self.out_ch) # [T, B, N, E, out_ch]

        return x_pred

class TimeConvODE_x(nn.Module):
    def __init__(self, in_ch, out_ch, modes, act, solver, rtol, atol, fourier_basis=None, 
                 no_fourier=False, no_ode=False, mode_interaction="no_interaction",
                 time_mode='none',
                 time_embedding_dim=16,
                 max_positions=5000):
        super(TimeConvODE_x, self).__init__()
        self.t_conv = SpectralConv1dODE_x(in_ch, out_ch, modes, solver, rtol, atol, fourier_basis, 
                                          no_fourier=no_fourier, no_ode=no_ode, mode_interaction=mode_interaction, 
                                          time_mode=time_mode, time_embedding_dim=time_embedding_dim, max_positions=max_positions)
        self.act = act 

        self.no_fourier = no_fourier
        self.no_ode = no_ode 
        self.time_mode = time_mode
        self.time_embedding_dim = time_embedding_dim
        self.max_positions = max_positions

    def forward(self, x, times, U_batch=None):
        h = self.t_conv(x, times, U_batch)
        out = self.act(h)
        # pls use broadcasting for the following line to work 
        # add one more dimension to the tensor to make it align with [B, N, T, E, 2] 
        x = x.unsqueeze(0) # [B, N, 1, E, 2] 
        return x + out


