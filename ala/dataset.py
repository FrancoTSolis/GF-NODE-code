import numpy as np
import torch
import os
from torch_geometric.utils import to_dense_adj
import mdshare
import mdtraj as md

# reimplement the get_ala2_trajs with the local filenames
def get_ala2_trajs(path=None, scale=False):
    topology = os.path.join(path, "alanine-dipeptide-nowater.pdb")
    filenames = [os.path.join(path, f"alanine-dipeptide-{i}-250ns-nowater.xtc") for i in range(3)] 
    trajs = [md.load_xtc(fn, topology) for fn in filenames]
    trajs = [t.center_coordinates().xyz for t in trajs]
    if scale:
        std = 0.1661689 
        trajs = [t / std for t in trajs]
    return trajs


def get_ala2_atom_numbers(distinguish=False):
    # fmt: off
    ALA2ATOMNUMBERS = [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1]
    # fmt: on
    atom_numbers = torch.tensor(  # pylint: disable=not-callable
        list(range(len(ALA2ATOMNUMBERS))) if distinguish else ALA2ATOMNUMBERS,
        dtype=torch.long,
    )
    return atom_numbers


class AlanineDataset:
    """
    Alanine Dipeptide Dataset adapted from MD17 style
    """
    def __init__(self, partition, max_samples, delta_frame, data_dir, molecule_type, 
                 num_timesteps=8, uneven_sampling=False, internal_seed=None,
                 scale=False, distinguish=False):
        """
        Args:
            partition (str): 'train', 'val', or 'test'
            max_samples (int): Maximum number of samples to use
            delta_frame (int): Number of frames between start and end
            data_dir (str): Directory containing data
            num_timesteps (int): Number of intermediate timesteps to sample
            uneven_sampling (bool): Whether to sample timesteps unevenly
            internal_seed (int): Random seed for uneven sampling
            scale (bool): Whether to scale coordinates
            distinguish (bool): Whether to distinguish atom types
        """

        assert molecule_type == 'ala2', "Molecule type must be 'ala2'" 
        # Setup split ratios
        train_par, val_par, test_par = 0.8, 0.1, 0.1
        self.partition = partition
        self.delta_frame = delta_frame
        self.num_timesteps = num_timesteps
        self.uneven_sampling = uneven_sampling
        
        if self.uneven_sampling:
            self.internal_seed = internal_seed
            self.random_state = np.random.RandomState(self.internal_seed)

        # Load trajectories
        trajs = get_ala2_trajs(data_dir, scale=scale)
        # Concatenate all trajectories
        x = np.concatenate(trajs, axis=0)  # Shape: [n_frames, n_atoms, 3]
        # print("x.shape:", x.shape) # (750000, 22, 3) # 750000 frames, 22 atoms, 3 dimensions 
        
        # Calculate velocities (assuming dt=1, will need scaling)
        dt = 1.0  # You may need to adjust this based on actual timestep
        v = (x[1:] - x[:-1]) / dt
        x = x[:-1]  # Remove last frame to match velocity length

        # Create train/val/test split
        np.random.seed(100)
        n_samples = x.shape[0] - self.delta_frame # 750000 - 3000 = 747000, 747000 frames
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_size = int(train_par * n_samples)
        val_size = int(val_par * n_samples)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Select partition
        if partition == 'train':
            st = train_idx
        elif partition == 'val':
            st = val_idx
        elif partition == 'test':
            st = test_idx
        else:
            raise ValueError(f"Invalid partition: {partition}")

        st = st[:max_samples]

        # Get atom numbers
        z = get_ala2_atom_numbers(distinguish=distinguish).numpy()
        self.n_node = len(z)

        # Get initial positions and velocities
        x_0, v_0 = x[st], v[st]

        # Sample intermediate frames
        if self.uneven_sampling:
            sampled_frames = np.zeros((len(st), self.num_timesteps), dtype=int)
            for i in range(len(st)):
                frame_0 = st[i]
                frame_T = st[i] + delta_frame
                frames_range = np.arange(frame_0 + 1, frame_T + 1)
                sampled_frames[i] = self.random_state.choice(
                    frames_range, size=self.num_timesteps, replace=False)
                sampled_frames[i].sort()
        else:
            frames_indices = np.arange(1, self.num_timesteps + 1)
            sampled_frames = st[:, None] + (delta_frame * frames_indices[None, :]) // self.num_timesteps

        # Calculate timeframes relative to start frames
        timeframes = sampled_frames - st[:, None]

        # Extract positions and velocities at sampled frames
        x_t = x[sampled_frames]
        v_t = v[sampled_frames]

        # Rearrange axes
        x_t = np.transpose(x_t, (0, 2, 1, 3))
        v_t = np.transpose(v_t, (0, 2, 1, 3))

        print(f'Got {x_0.shape[0]} samples!')

        # Build graph structure
        _lambda = 4.0 / 10  # Adjust cutoff distance as needed
        
        def d(_i, _j, _t):
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        # Create edges based on distance criterion
        atom_edges = torch.zeros(self.n_node, self.n_node).int()
        for i in range(self.n_node):
            for j in range(self.n_node):
                if i != j:
                    _d = d(i, j, 0)
                    if _d < _lambda:
                        atom_edges[i][j] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2

        print(atom_edges)

        # Create edge attributes
        edge_attr = []
        rows, cols = [], []
        for i in range(self.n_node):
            for j in range(self.n_node):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([z[i], z[j], 1])
                    if self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([z[i], z[j], 2])

        edges = [rows, cols]
        edge_attr = torch.Tensor(np.array(edge_attr))
        self.edge_attr = edge_attr
        self.edges = edges

        # Compute graph Fourier basis
        A = to_dense_adj(torch.LongTensor(self.edges), 
                        batch=torch.zeros(self.n_node).long())
        A = A.squeeze(0)

        # Compute degree matrix D
        deg = torch.sum(A, dim=1)
        D = torch.diag(deg)

        # Compute Laplacian L = D - A
        L = D - A

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        self.U = eigenvectors

        # Store data as tensors
        self.Z = torch.Tensor(z)
        self.x_0 = torch.Tensor(x_0)
        self.v_0 = torch.Tensor(v_0)
        self.x_t = torch.Tensor(x_t)
        self.v_t = torch.Tensor(v_t)
        self.mole_idx = torch.Tensor(z)
        self.timeframes = torch.Tensor(timeframes)

        # Get molecular configuration
        self.cfg = self.sample_cfg()

    def sample_cfg(self):
        """
        Define molecular configuration for alanine dipeptide
        """
        cfg = {}
        # Define stick bonds (you may need to adjust these based on actual molecular structure)
        # cfg['Stick'] = [] # [(0, 1), (1, 4), (4, 6), (6, 8), (8, 14), (14, 16)]
        cfg['Stick'] = [
            (1, 0),   # CH3 - HH31 in ACE
            (1, 2),   # CH3 - HH32 in ACE
            (1, 3),   # CH3 - HH33 in ACE
            (1, 4),   # CH3 - C in ACE
            (4, 5),   # C - O in ACE
            (4, 6),   # C - N between ACE and ALA
            (6, 7),   # N - H in ALA
            (6, 8),   # N - CA in ALA
            (8, 9),   # CA - HA in ALA
            (8, 10),  # CA - CB in ALA
            (8, 14),  # CA - C in ALA
            (10, 11), # CB - HB1 in ALA
            (10, 12), # CB - HB2 in ALA
            (10, 13), # CB - HB3 in ALA
            (14, 15), # C - O in ALA
            (14, 16), # C - N between ALA and NME
            (16, 17), # N - H in NME
            (16, 18), # N - CH3 in NME
            (18, 19), # CH3 - HH31 in NME
            (18, 20), # CH3 - HH32 in NME
            (18, 21)  # CH3 - HH33 in NME
        ]        
        # Find isolated atoms
        cur_selected = []
        for stick in cfg['Stick']:
            cur_selected.extend(stick)
        cfg['Isolated'] = [[i] for i in range(self.n_node) if i not in cur_selected]
        
        if len(cfg['Isolated']) == 0:
            cfg.pop('Isolated')
            
        return cfg

    def __getitem__(self, i):
        cfg = self.cfg
        edge_attr = self.edge_attr
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        edges = self.edges

        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            if 'Stick' in cfg:
                for stick in cfg['Stick']:
                    if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
                        stick_ind[m] = 1

        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)
        cfg = {k: torch.from_numpy(np.array(cfg[k])) for k in cfg}

        return (self.x_0[i], self.v_0[i], edge_attr, self.mole_idx.unsqueeze(-1), 
                self.x_t[i], self.v_t[i], self.Z.unsqueeze(-1), 
                self.timeframes[i], self.U, cfg)

    def __len__(self):
        return len(self.x_0)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg
    
if __name__ == "__main__":
    # load the dataset, and visualize the magnitude of the positions and velocities 
    dataset = AlanineDataset(partition='train', max_samples=500, delta_frame=3000, molecule_type='ala2', data_dir='ala', num_timesteps=8, uneven_sampling=False, internal_seed=None, scale=False, distinguish=False)
    # visualize the magnitude of the positions and velocities
    x_0 = dataset.x_0.numpy()
    v_0 = dataset.v_0.numpy()
    print(x_0.shape, v_0.shape)
    print(np.max(x_0), np.min(x_0))
    print(np.max(v_0), np.min(v_0))
    # plot the positions and velocities
    import matplotlib.pyplot as plt 
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].hist(x_0.flatten(), bins=100)
    axs[1].hist(v_0.flatten(), bins=100)
    plt.show()
