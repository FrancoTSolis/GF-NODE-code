import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchdiffeq import odeint 
from torch_geometric.utils import to_dense_adj 

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
    def __init__(self, in_ch, out_ch, modes1):
        super(ODEFunction_real, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = (1 / (in_ch * out_ch))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.modes1, in_ch, out_ch, dtype=torch.float)
        )

    def forward(self, t, x_hat):
        # x_hat: [M, B, in_ch]
        # weights1: [M, in_ch, out_ch]
        out_hat = torch.einsum("mni,mio->mno", x_hat, self.weights1)
        return out_hat

class SpectralConv1dODE(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, solver, rtol, atol, fourier_basis=None, 
                 no_fourier=False, no_ode=False):
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

        assert not (no_fourier and no_ode), "no_fourier and no_ode cannot be True at the same time" 

        
        if no_fourier is True:
            self.ode_func = ODEFunction_real(in_ch, out_ch, modes1) 
        elif fourier_basis == "linear": 
            self.ode_func = ODEFunction_complex(in_ch, out_ch, modes1)
        elif fourier_basis == "graph":
            self.ode_func = ODEFunction_real(in_ch, out_ch, modes1) 
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
                 no_fourier=False, no_ode=False):
        super(TimeConvODE, self).__init__()
        self.t_conv = SpectralConv1dODE(in_ch, out_ch, modes, solver, rtol, atol, fourier_basis, 
                                        no_fourier=no_fourier, no_ode=no_ode)
        self.act = act

        self.no_fourier = no_fourier
        self.no_ode = no_ode

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

class ODEFunction_real_x(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(ODEFunction_real_x, self).__init__()
        self.in_ch = in_ch  # Input channels (e.g., positions and velocities)
        self.out_ch = out_ch  # Output channels (should match in_ch)
        self.modes1 = modes1

        self.scale = (1 / (in_ch * out_ch))
        # Weights shape: [modes1, in_ch, out_ch]
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.modes1, self.in_ch, self.out_ch, dtype=torch.float)
        )

    def forward(self, t, x_hat):
        # x_hat shape: [M, B, E, in_ch]
        # weights1 shape: [M, in_ch, out_ch]
        # Output shape: [M, B, E, out_ch]
        out_hat = torch.einsum('mbei,mio->mbeo', x_hat, self.weights1)
        return out_hat


class SpectralConv1dODE_x(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, solver, rtol, atol, fourier_basis=None, 
                 no_fourier=False, no_ode=False):
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

        assert not (no_fourier and no_ode), "no_fourier and no_ode cannot be True at the same time" 

        if no_fourier is True:
            self.ode_func = ODEFunction_real_x(in_ch, out_ch, modes1) 
        elif fourier_basis == "linear": 
            self.ode_func = ODEFunction_complex_x(in_ch, out_ch, modes1)
        elif fourier_basis == "graph":
            self.ode_func = ODEFunction_real_x(in_ch, out_ch, modes1)
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
                 no_fourier=False, no_ode=False):
        super(TimeConvODE_x, self).__init__()
        self.t_conv = SpectralConv1dODE_x(in_ch, out_ch, modes, solver, rtol, atol, fourier_basis, 
                                          no_fourier=no_fourier, no_ode=no_ode)
        self.act = act 

        self.no_fourier = no_fourier
        self.no_ode = no_ode 

    def forward(self, x, times, U_batch=None):
        h = self.t_conv(x, times, U_batch)
        out = self.act(h)
        # pls use broadcasting for the following line to work 
        # add one more dimension to the tensor to make it align with [B, N, T, E, 2] 
        x = x.unsqueeze(0) # [B, N, 1, E, 2] 
        return x + out


