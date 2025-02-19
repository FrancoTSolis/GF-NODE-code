from torch import nn
import torch
import torch.nn.functional as F
from model.basic import EGNN_Layer 
from model.layer_no_ode import TimeConvODE, TimeConvODE_x, get_timestep_embedding, GraphODEBlock 
from model.basic import GraphSAGE_Layer, GCN_Layer 

###############################################################################
# The FourierMD class (updated with gnn_ablation_mode)
###############################################################################
class FourierMD(nn.Module):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False, num_modes=2, num_timesteps=8, time_emb_dim=32, 
                 num_atoms=5, solver='dopri5', rtol=1e-3, atol=1e-4, delta_frame=None, 
                 fourier_basis=None, 
                 no_fourier=False, no_ode=False,
                 use_vae=False,
                 gnn_ablation_mode='EGNN'):  
        super(FourierMD, self).__init__()
        self.encode_layers = nn.ModuleList() 
        self.decode_layers = nn.ModuleList() 
        self.n_layers = n_layers
        self.num_modes = num_modes 
        self.device = device 
        self.num_timesteps = num_timesteps 
        self.time_emb_dim = time_emb_dim 
        self.encode_hidden_nf = hidden_nf 
        self.decode_hidden_nf = hidden_nf + time_emb_dim 

        self.num_atoms = num_atoms 
        self.solver = solver
        self.rtol = rtol
        self.atol = atol 
        self.delta_frame = delta_frame 

        self.fourier_basis = fourier_basis 
        if no_fourier: 
            assert fourier_basis is None, "fourier_basis must be None if no_fourier is True"
        else: 
            assert (fourier_basis == 'linear' or fourier_basis == 'graph'), "fourier_basis must be either 'linear' or 'graph'"

        self.no_fourier = no_fourier 
        self.no_ode = no_ode 
        self.use_vae = use_vae 
        self.gnn_ablation_mode = gnn_ablation_mode  # store the chosen ablation mode

        # input feature mapping 
        self.embedding = nn.Linear(in_node_nf, self.encode_hidden_nf)

        # Depending on gnn_ablation_mode, choose the layer class
        # We'll define a small helper:
        def make_graph_layer(in_edge_nf, hidden_nf, activation, with_v, flat, norm):
            if self.gnn_ablation_mode.lower() == 'egnn':
                return EGNN_Layer(
                    in_edge_nf, 
                    hidden_nf, 
                    activation=activation, 
                    with_v=with_v, 
                    flat=flat, 
                    norm=norm
                )
            elif self.gnn_ablation_mode.lower() in ['sage', 'graphsage']:
                return GraphSAGE_Layer(
                    in_edge_nf,
                    hidden_nf,
                    activation=activation,
                    with_v=with_v,
                    flat=flat
                )
            elif self.gnn_ablation_mode.lower() == 'gcn':
                return GCN_Layer(
                    in_edge_nf,
                    hidden_nf,
                    activation=activation,
                    with_v=with_v,
                    flat=flat
                )
            else:
                raise ValueError(f"Unknown gnn_ablation_mode: {self.gnn_ablation_mode}")

        # Build encoder layers
        for i in range(self.n_layers):
            layer = make_graph_layer(in_edge_nf, self.encode_hidden_nf, activation, with_v, flat, norm)
            self.encode_layers.append(layer) 
        
        if self.use_vae:
            # Define layers to compute mean and logvar for h
            self.fc_mu_h = nn.Linear(self.encode_hidden_nf, self.encode_hidden_nf)
            self.fc_logvar_h = nn.Linear(self.encode_hidden_nf, self.encode_hidden_nf)
            # Similarly for x
            self.fc_mu_x = nn.Linear(3, 3)
            self.fc_logvar_x = nn.Linear(3, 3)

        self.freq_conv = TimeConvODE(self.encode_hidden_nf, self.encode_hidden_nf, num_modes, activation, 
                                     solver, rtol, atol, fourier_basis=fourier_basis, 
                                     no_fourier=no_fourier, no_ode=no_ode) 
        self.freq_conv_x = TimeConvODE_x(2, 2, num_modes, activation, 
                                         solver, rtol, atol, fourier_basis=fourier_basis, 
                                         no_fourier=no_fourier, no_ode=no_ode) 

        # Build decoder layers
        for i in range(self.n_layers):
            layer = make_graph_layer(in_edge_nf, self.decode_hidden_nf, activation, with_v, flat, norm)
            self.decode_layers.append(layer) 

        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, v=None, loc_mean=None, timeframes=None, U_batch=None):
        
        N, T = self.num_atoms, self.num_timesteps 
        num_nodes = h.shape[0] 
        num_edges = edge_index[0].shape[0]
        B = num_nodes // N 

        h = self.embedding(h)
        for i in range(self.n_layers):
            x, v, h = self.encode_layers[i](x, h, edge_index, edge_fea, v=v)

        if self.use_vae:
            # For h
            mu_h = self.fc_mu_h(h)
            logvar_h = self.fc_logvar_h(h)
            std_h = torch.exp(0.5 * logvar_h)
            eps_h = torch.randn_like(std_h)
            h = mu_h + eps_h * std_h  # Reparameterization trick

            # For x
            mu_x = self.fc_mu_x(x)
            logvar_x = self.fc_logvar_x(x)
            std_x = torch.exp(0.5 * logvar_x)
            eps_x = torch.randn_like(std_x)
            x = mu_x + eps_x * std_x

            # Compute KL divergence
            kld_h = -0.5 * torch.sum(1 + logvar_h - mu_h.pow(2) - logvar_h.exp(), dim=1)
            kld_x = -0.5 * torch.sum(1 + logvar_x - mu_x.pow(2) - logvar_x.exp(), dim=1)
            kld_loss = (kld_h + kld_x).mean()  # Average over batch
        else:
            kld_loss = torch.tensor(0.0, device=x.device)

        # normalize the timeframes to (0, 1], important for numerical stability of ode solver
        normalized_timeframes = timeframes / self.delta_frame 

        h = self.freq_conv(h.view((B, N, -1)), normalized_timeframes, U_batch) # [T, B, N, E]

        x_translated = x - loc_mean
        X = torch.stack((x_translated, v), dim=-1)
        temp = self.freq_conv_x(X.view(B, N, 3, 2), normalized_timeframes, U_batch) # [T, B, N, 3, 2] 
        loc_mean = loc_mean.repeat(T, 1) 
        x = temp[..., 0].contiguous().view(T * num_nodes, 3) + loc_mean
        v = temp[..., 1].contiguous().view(T * num_nodes, 3)

        # Adjust time embeddings
        time_emb = get_timestep_embedding(
            timeframes,
            embedding_dim=self.time_emb_dim,
            max_positions=10000
        )  # Shape: [B, T, H_t]
        time_emb = time_emb.unsqueeze(2).repeat(1, 1, N, 1)  # Shape: [B, T, N, H_t]
        time_emb = time_emb.swapaxes(0, 1).contiguous().view(T, num_nodes, -1)  # Shape: [T, BN, H_t]

        h = h.contiguous().view(T, num_nodes, -1)  # [T, BN, H]
        h = torch.cat((h, time_emb), dim=-1)  # [T, BN, H+H_t]
        h = h.view(-1, h.shape[-1])  # [T*BN, H+H_t]

        cumsum = torch.arange(0, T).to(self.device) * num_nodes
        cumsum_nodes = cumsum.repeat_interleave(num_nodes, dim=0)
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)

        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges
        edge_index = [edges_0, edges_1]
        edge_fea = edge_fea.repeat(T, 1) 

        for i in range(self.n_layers): 
            x, v, h = self.decode_layers[i](x, h, edge_index, edge_fea, v=v)

        return x, v, h, kld_loss 

        
# test FourierMD 
if __name__ == "__main__": 
    B, N, E = 11, 78, 64  # Example batch size, particles, and encoding size
    T = 10  # Number of time points
    x = torch.randn(B * N, 3)  # Example input tensor
    h = torch.randn(B * N, 64)  # Example input tensor
    edge_index = torch.randint(0, B * N, (2, 100), device=x.device)
    edge_fea = torch.randn(100, 2)
    v = torch.randn(B * N, 3)  # Example input tensor
    loc_mean = torch.randn(B * N, 3)  # Example input tensor

    model = FourierMD(n_layers=2, in_node_nf=64, in_edge_nf=2, hidden_nf=64, activation=nn.LeakyReLU(), device=x.device, with_v=True,
                      flat=False, norm=False, num_modes=3, num_timesteps=T, time_emb_dim=32, num_atoms=N, solver='dopri5', rtol=1e-3, atol=1e-4)
    output = model(x, h, edge_index, edge_fea, v=v, loc_mean=loc_mean)  # Output shape: [B, N, T, E]
    print(output[0].shape) # torch.Size([8580, 3])
    print(output[1].shape) # torch.Size([8580, 3])
    print(output[2].shape) # torch.Size([8580, 96])
    print("Success")        
    
    

    

    



    
    


        

class GraphODEMD(nn.Module):
    """
    An alternative to FourierMD that:
      - (Optionally) uses VAE to sample initial x, h
      - Integrates hidden states via GNN-based ODE (no Fourier transform)
      - Decodes each time-slice with a simple MLP. 
    """
    def __init__(
        self,
        n_layers,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device='cpu',
        with_v=False,
        flat=False,
        norm=False,
        num_timesteps=8,
        num_atoms=5,
        solver='dopri5',
        rtol=1e-3,
        atol=1e-4,
        delta_frame=None,
        use_vae=False,
        gnn_ablation_mode='EGNN'
    ):
        super(GraphODEMD, self).__init__()
        self.n_layers = n_layers
        self.num_timesteps = num_timesteps
        self.num_atoms = num_atoms
        self.delta_frame = delta_frame
        self.device = device
        self.use_vae = use_vae

        # 1) Build an encoder GNN (stacked layers) to produce node embeddings
        self.encoder_layers = nn.ModuleList()
        self.hidden_nf = hidden_nf
        self.with_v = with_v
        self.flat = flat
        self.norm = norm
        self.gnn_ablation_mode = gnn_ablation_mode.lower()

        def make_graph_layer(in_edge_nf, hidden_nf, activation):
            if self.gnn_ablation_mode == 'egnn':
                return EGNN_Layer(
                    in_edge_nf, hidden_nf, 
                    activation=activation, with_v=with_v, flat=flat, norm=norm
                )
            elif self.gnn_ablation_mode in ['sage', 'graphsage']:
                return GraphSAGE_Layer(
                    in_edge_nf, hidden_nf, 
                    activation=activation, with_v=with_v, flat=flat
                )
            elif self.gnn_ablation_mode == 'gcn':
                return GCN_Layer(
                    in_edge_nf, hidden_nf, 
                    activation=activation, with_v=with_v, flat=flat
                )
            else:
                raise ValueError(f"Unknown gnn_ablation_mode: {self.gnn_ablation_mode}")

        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)

        for _ in range(self.n_layers):
            layer = make_graph_layer(in_edge_nf, hidden_nf, activation)
            self.encoder_layers.append(layer)

        # If VAE is used
        if self.use_vae:
            self.fc_h_mu = nn.Linear(hidden_nf, hidden_nf)
            self.fc_h_logvar = nn.Linear(hidden_nf, hidden_nf)

            self.fc_x_mu = nn.Linear(3, 3)
            self.fc_x_logvar = nn.Linear(3, 3)

        # 2) GNN ODE Block – we'll use a single GNN layer for demonstration,
        #    or you can define a small stack. For now, we reuse one layer from the same
        #    ablation mode. If you prefer multiple layers in the ODE func, pass an nn.Sequential, etc.
        self.ode_gnn = make_graph_layer(in_edge_nf, hidden_nf, activation)
        self.ode_block = GraphODEBlock(self.ode_gnn, solver=solver, rtol=rtol, atol=atol)

        # 3) Simple Decoder MLP for each time step
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, 3)  # decode to 3D positions, or 3D velocity, etc.
        )

        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, v=None, loc_mean=None, timeframes=None, U_batch=None):
        """
        x: [B*N, 3], positions
        h: [B*N, in_node_nf], node features
        edge_index, edge_fea: graph structure
        v: [B*N, 3], velocities (unused here, except if your GNN layer needs it)
        loc_mean: [B*N, 3], optional translation
        timeframes: [B], or [B, T], depending on usage
        U_batch: not used here (we skip Fourier)
        """
        B = x.shape[0] // self.num_atoms
        # 1) Encode: run h through encoder GNN
        h = self.embedding(h)
        for layer in self.encoder_layers:
            x, v, h = layer(x, h, edge_index, edge_fea, v=v)

        # If VAE, reparameterize to get new h and x
        kld_loss = torch.tensor(0.0, device=h.device)
        if self.use_vae:
            # Node feats
            mu_h = self.fc_h_mu(h)
            logvar_h = self.fc_h_logvar(h)
            std_h = torch.exp(0.5 * logvar_h)
            eps_h = torch.randn_like(std_h)
            h = mu_h + eps_h * std_h

            # Positions
            mu_x = self.fc_x_mu(x)
            logvar_x = self.fc_x_logvar(x)
            std_x = torch.exp(0.5 * logvar_x)
            eps_x = torch.randn_like(std_x)
            x = mu_x + eps_x * std_x

            # KLD
            kld_h = -0.5 * torch.sum(1 + logvar_h - mu_h.pow(2) - logvar_h.exp(), dim=1)
            kld_x = -0.5 * torch.sum(1 + logvar_x - mu_x.pow(2) - logvar_x.exp(), dim=1)
            kld_loss = (kld_h + kld_x).mean()

        # 2) Integrate with a GNN-based ODE
        # We interpret h as the state. Provide edge data to the ODE block.
        self.ode_block.set_graph(edge_index, edge_fea)

        # Suppose timeframes is shape [B], meaning each sample has a single end time. 
        # Or shape [B, T]. For demonstration, let's do T = self.num_timesteps steps linearly from 0..1
        t_span = torch.linspace(0, 1, steps=self.num_timesteps, device=self.device)

        # We shape h as [B*N, hidden_nf], no per-sample difference in initial time -> apply ODE
        h0 = h
        ode_sol = self.ode_block(h0, t_span)  # [T, B*N, hidden_nf]

        # 3) Decode each time step into positions (or something)
        # We produce x(t) from h(t). 
        # For simplicity, we skip velocity in decode. Adapt as you like.
        x_list = []
        for i in range(self.num_timesteps):
            h_t = ode_sol[i]
            x_decoded = self.decoder(h_t)  # [B*N, 3]
            # Optionally add back loc_mean if relevant
            if loc_mean is not None:
                x_decoded = x_decoded + loc_mean
            x_list.append(x_decoded)

        # Stack them: shape [T*B*N, 3], or keep them separate
        x_final = torch.stack(x_list, dim=0).reshape(-1, 3)  # if you like [T*B*N, 3]
        # We don't do anything with velocity here, but you could similarly decode velocity.

        # Return (positions, velocity dummy, hidden states, kld loss)
        # Velocity is unused => return zeros if needed
        dummy_v = torch.zeros_like(x_final, device=x.device)
        # If you prefer returning the entire trajectory, you can do so. 
        return x_final, dummy_v, ode_sol[-1], kld_loss  # final time hidden state

        
# test GraphODEMD 
if __name__ == "__main__": 
    B, N, E = 11, 78, 64  # Example batch size, particles, and encoding size
    T = 10  # Number of time points
    x = torch.randn(B * N, 3)  # Example input tensor
    h = torch.randn(B * N, 64)  # Example input tensor
    edge_index = torch.randint(0, B * N, (2, 100), device=x.device)
    edge_fea = torch.randn(100, 2)
    v = torch.randn(B * N, 3)  # Example input tensor
    loc_mean = torch.randn(B * N, 3)  # Example input tensor

    model = GraphODEMD(n_layers=2, in_node_nf=64, in_edge_nf=2, hidden_nf=64, activation=nn.LeakyReLU(), device=x.device, with_v=True,
                      flat=False, norm=False, num_timesteps=T, num_atoms=N, solver='dopri5', rtol=1e-3, atol=1e-4)
    output = model(x, h, edge_index, edge_fea, v=v, loc_mean=loc_mean)  # Output shape: [B, N, T, E]
    print(output[0].shape) # torch.Size([8580, 3])
    print(output[1].shape) # torch.Size([8580, 3])
    print(output[2].shape) # torch.Size([8580, 96])
    print("Success")        
    
    

    

    



    
    


        