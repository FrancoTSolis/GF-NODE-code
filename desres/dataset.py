import numpy as np
import torch
import os
import csv
import MDAnalysis as mda
from torch_geometric.utils import to_dense_adj

class DESRESDataset:
    """
    DESRES Trajectory Dataset
    """
    def __init__(self, partition, max_samples, delta_frame, data_dir, molecule_type, 
                 num_timesteps=8, uneven_sampling=False, internal_seed=None,
                 time_ref=None):
        """
        Args:
            partition (str): 'train', 'val', or 'test'
            max_samples (int): Maximum number of samples to use
            delta_frame (int): Number of frames between start and end
            data_dir (str): Directory containing data
            molecule_type (str): Molecule type (e.g., '1FME')
            num_timesteps (int): Number of intermediate timesteps to sample
            uneven_sampling (bool): Whether to sample timesteps unevenly
            internal_seed (int): Random seed for uneven sampling
            time_ref (int): Reference time step for normalization
        """
        # Setup split ratios
        train_par, val_par, test_par = 0.8, 0.1, 0.1
        self.partition = partition
        self.delta_frame = delta_frame
        self.num_timesteps = num_timesteps
        self.uneven_sampling = uneven_sampling
        
        if self.uneven_sampling:
            self.internal_seed = internal_seed
            self.random_state = np.random.RandomState(self.internal_seed)

        # Construct paths based on molecule name
        protein_folder = f"{data_dir}/{molecule_type}-0-protein"
        pdb_file = os.path.join(protein_folder, f"{molecule_type}-0-protein.pdb")
        csv_file = os.path.join(protein_folder, f"{molecule_type}-0-protein_times.csv")
        
        print(f"Loading DESRES data for: {molecule_type}")
        print(f"PDB file: {pdb_file}")
        print(f"CSV catalog file: {csv_file}")
        
        # Read catalog from CSV file
        catalog = []
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) == 2:
                    start_time_ps = float(row[0])
                    filename = os.path.join(protein_folder, row[1])
                    catalog.append((start_time_ps, filename))
        
        print(f"Found {len(catalog)} trajectory segments")
        
        # Extract just the filenames (in the correct order)
        traj_files = [fn for _, fn in catalog]
        
        # Load trajectory with MDAnalysis
        u = mda.Universe(pdb_file, *traj_files, dt=200.0)
        
        print(f"Loaded trajectory with {len(u.trajectory)} frames and {len(u.atoms)} atoms")
        
        # Filter to only include heavy atoms for efficiency
        heavy_atoms = u.select_atoms("not name H*")
        n_atoms = len(heavy_atoms)
        self.n_node = n_atoms
        print(f"Selected {n_atoms} heavy atoms")
        
        # Extract atom information
        atom_names = [atom.name for atom in heavy_atoms]
        atom_types = [atom.type for atom in heavy_atoms]
        residues = [atom.resname for atom in heavy_atoms]
        
        # Create atom type mapping for one-hot encoding
        unique_types = list(set(atom_types))
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        z = np.array([type_to_idx[t] for t in atom_types])
        
        # Collect positions and calculate velocities
        n_frames = len(u.trajectory)
        positions = np.zeros((n_frames, n_atoms, 3))
        
        print(f"Extracting coordinates from trajectory...")
        for i, ts in enumerate(u.trajectory):
            if i % 1000 == 0:
                print(f"Processing frame {i}/{n_frames}")
            positions[i] = heavy_atoms.positions
        
        # Calculate velocities (v[i] = (x[i+1] - x[i]) / dt)
        velocities = (positions[1:] - positions[:-1]) / u.trajectory.dt
        positions = positions[:-1]  # Remove last frame to match velocity length
        
        # Create train/val/test split
        np.random.seed(100)
        n_samples = positions.shape[0] - self.delta_frame
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
        
        # Get initial positions and velocities
        x_0, v_0 = positions[st], velocities[st]
        
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
        x_t = positions[sampled_frames]
        v_t = velocities[sampled_frames]
        
        # Rearrange axes for compatibility with model
        x_t = np.transpose(x_t, (0, 2, 1, 3))
        v_t = np.transpose(v_t, (0, 2, 1, 3))
        
        print(f'Got {x_0.shape[0]} samples for {partition}!')
        
        # Build graph structure
        # Use distance-based cutoff for edges
        cutoff = 8.0  # Angstroms, adjust as needed for protein structure
        
        # Create adjacency matrix based on distance criterion
        # Use initial frame for edge determination
        frame_idx = 0  # Use first frame for graph structure
        atom_edges = torch.zeros(self.n_node, self.n_node).int()
        
        # Calculate all pairwise distances
        pos = positions[frame_idx]
        distances = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    distances[i, j] = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                    if distances[i, j] < cutoff:
                        atom_edges[i, j] = 1
        
        # Create second-order connectivity (neighbors of neighbors)
        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        
        # Create edge attributes
        edge_attr = []
        rows, cols = [], []
        for i in range(self.n_node):
            for j in range(self.n_node):
                if i != j:
                    if self.atom_edge[i, j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([z[i], z[j], 1])
                    if self.atom_edge2[i, j]:
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
        self.cfg = self.sample_cfg(residues, atom_names)
    
    def sample_cfg(self, residues, atom_names):
        """
        Define molecular configuration for protein structure
        This creates bond information based on standard protein connectivity
        """
        cfg = {}
        
        # Create a mapping from (residue_idx, atom_name) to atom index
        residue_indices = [int(res.split()[-1]) if ' ' in res else 0 for res in residues]
        atom_map = {}
        for i, (res_idx, atom_name) in enumerate(zip(residue_indices, atom_names)):
            atom_map[(res_idx, atom_name)] = i
        
        # Define stick bonds for protein backbone
        sticks = []
        prev_ca = None
        prev_c = None
        
        for res_idx in sorted(set(residue_indices)):
            # Add backbone bonds within residue
            if (res_idx, 'N') in atom_map and (res_idx, 'CA') in atom_map:
                sticks.append((atom_map[(res_idx, 'N')], atom_map[(res_idx, 'CA')]))
            
            if (res_idx, 'CA') in atom_map and (res_idx, 'C') in atom_map:
                sticks.append((atom_map[(res_idx, 'CA')], atom_map[(res_idx, 'C')]))
            
            # Connect to previous residue
            if prev_c is not None and (res_idx, 'N') in atom_map:
                sticks.append((prev_c, atom_map[(res_idx, 'N')]))
            
            # Save reference to this residue's atoms for next iteration
            if (res_idx, 'CA') in atom_map:
                prev_ca = atom_map[(res_idx, 'CA')]
            if (res_idx, 'C') in atom_map:
                prev_c = atom_map[(res_idx, 'C')]
        
        cfg['Stick'] = sticks
        
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
    # Test the dataset
    dataset = DESRESDataset(
        partition='train', 
        max_samples=100, 
        delta_frame=100, 
        data_dir='desres', 
        molecule_type='1FME', 
        num_timesteps=8, 
        uneven_sampling=False, 
        internal_seed=None
    )
    
    # Print dataset statistics
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of atoms: {dataset.n_node}")
    print(f"Number of edges: {len(dataset.edges[0])}")
    
    # Sample and print item
    sample = dataset[0]
    print(f"Sample x_0 shape: {sample[0].shape}")
    print(f"Sample v_0 shape: {sample[1].shape}")
    print(f"Sample x_t shape: {sample[4].shape}")
    print(f"Sample v_t shape: {sample[5].shape}")
    print(f"Sample timeframes shape: {sample[7].shape}") 