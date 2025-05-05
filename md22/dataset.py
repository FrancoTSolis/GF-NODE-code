import numpy as np
import torch
import pickle as pkl
import os
from torch_geometric.utils import to_dense_adj


class MD22Dataset():
    """
    MD22 Dataset base class
    """

    def __init__(self, partition, max_samples, delta_frame, data_dir, molecule_type):
        raise NotImplementedError() 

    def sample_cfg(self):
        """
        Kinematics Decomposition for molecules in MD22
        """
        cfg = {}
        if self.molecule_type == 'buckyball-catcher':
            cfg['Stick'] = [(0, 1), (2, 3), (4, 5)]
        elif self.molecule_type == 'AT-AT-CG-CG':
            cfg['Stick'] = [(0, 2), (1, 3), (5, 6)]
        elif self.molecule_type == 'AT-AT':
            cfg['Stick'] = [(0, 1), (2, 3)]
        elif self.molecule_type == 'stachyose':
            cfg['Stick'] = [(1, 2), (3, 4)]
        elif self.molecule_type == 'double-walled_nanotube':
            cfg['Stick'] = [(0, 1), (2, 3), (4, 5)]
        elif self.molecule_type == 'Ac-Ala3-NHMe':
            cfg['Stick'] = [(0, 1), (2, 3), (4, 5)]
        elif self.molecule_type == 'DHA':
            cfg['Stick'] = [(0, 1), (2, 3), (4, 5)]
        else:
            raise NotImplementedError()
        
        cur_selected = []
        for _ in cfg['Stick']:
            cur_selected.append(_[0])
            cur_selected.append(_[1])
        cfg['Isolated'] = [[_] for _ in range(self.n_node) if _ not in cur_selected]
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
            if 'Hinge' in cfg:
                for hinge in cfg['Hinge']:
                    if (row, col) in [(hinge[0], hinge[1]), (hinge[1], hinge[0]), (hinge[0], hinge[2]), (hinge[2], hinge[0])]:
                        stick_ind[m] = 2
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)  # [edge, 2]
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}

        return self.x_0[i], self.v_0[i], edge_attr, self.mole_idx.unsqueeze(-1), self.x_t[i], self.v_t[i], self.Z.unsqueeze(-1), self.timeframes[i], self.U, cfg

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
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


class MD22DynamicsDataset(MD22Dataset):
    """
    MD22 Dynamics Dataset
    """
    def __init__(self, partition, max_samples, delta_frame, data_dir, molecule_type, num_timesteps=8, 
                 uneven_sampling=False, internal_seed=None, time_ref=None):
        # setup a split, tentative setting
        train_par, val_par, test_par = 0.1, 0.05, 0.05
        full_dir = os.path.join(data_dir, 'md22_' + molecule_type + '.npz')
        split_dir = os.path.join(data_dir, molecule_type + '_split.pkl')
        data = np.load(full_dir)
        self.partition = partition
        self.molecule_type = molecule_type

        self.delta_frame = delta_frame 
        self.num_timesteps = num_timesteps 
        self.uneven_sampling = uneven_sampling # if True, sample timesteps unevenly
        self.time_ref = time_ref  # store the training delta_frame if provided
        if self.uneven_sampling: 
            self.internal_seed = internal_seed # seed for the random generator 
            self.random_state = np.random.RandomState(self.internal_seed)

        # MD22 data has the same structure as MD17
        x = data['R']
        v = x[1:] - x[:-1]
        x = x[:-1]

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except:
            print('Generating split for MD22 dataset...')
            np.random.seed(100)

            # For MD22, use the same splitting approach as MD17
            # but ensure we don't go out of bounds with smaller datasets
            total_frames = x.shape[0]
            buffer = min(10000, int(total_frames * 0.1))  # Use 10% as buffer or 10000, whichever is smaller
            
            # Use the middle part of the trajectory
            _x = x[buffer: -buffer] if buffer < total_frames // 2 else x
            
            # Create indices for train/val/test
            available_indices = np.arange(_x.shape[0])
            
            # Select fixed amounts for training/validation/testing as requested
            # or use percentages for smaller datasets
            if len(available_indices) > 5000:  # If dataset is large enough
                train_size = min(500, int(train_par * _x.shape[0]))
                val_size = min(2000, int(val_par * _x.shape[0]))
                test_size = min(2000, int(test_par * _x.shape[0]))
            else:
                # For smaller datasets, use percentages
                train_size = int(train_par * _x.shape[0])
                val_size = int(val_par * _x.shape[0])
                test_size = int(test_par * _x.shape[0])

            # Ensure we don't request more samples than available
            total_requested = train_size + val_size + test_size
            if total_requested > len(available_indices):
                # Scale down proportionally
                scale = len(available_indices) / total_requested
                train_size = max(int(train_size * scale), 1)
                val_size = max(int(val_size * scale), 1)
                test_size = max(int(test_size * scale), 1)
            
            # Sample without replacement
            train_idx = np.random.choice(available_indices, size=train_size, replace=False)
            flag = np.zeros(_x.shape[0])
            for _ in train_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            val_idx = np.random.choice(rest, size=val_size, replace=False)
            for _ in val_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            test_idx = np.random.choice(rest, size=test_size, replace=False)

            # Add the buffer offset back to get the original indices
            if buffer < total_frames // 2:
                train_idx += buffer
                val_idx += buffer
                test_idx += buffer

            split = (train_idx, val_idx, test_idx)

            with open(split_dir, 'wb') as f:
                pkl.dump(split, f)
            print(f'Generated and saved split! Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

        if partition == 'train':
            st = split[0]
        elif partition == 'val':
            st = split[1]
        elif partition == 'test':
            st = split[2]
        else:
            raise NotImplementedError()

        st = st[:max_samples]

        z = data['z']
        print('mol idx:', z)
        
        # In MD22, we don't filter atoms by atomic number
        # Use all atoms in the molecule
        x_0, v_0 = x[st], v[st]  # Initial positions and velocities at start frames

        if self.uneven_sampling:
            # Initialize an array to hold sampled frames for each data point
            sampled_frames = np.zeros((len(st), self.num_timesteps), dtype=int)
            for i in range(len(st)):
                frame_0 = st[i]
                frame_T = min(st[i] + delta_frame, x.shape[0] - 1)  # Ensure we don't exceed dataset bounds
                # Create a range of frames to sample from
                frames_range = np.arange(frame_0 + 1, frame_T + 1)
                # Check if we have enough frames
                if len(frames_range) < self.num_timesteps:
                    # If not enough frames, use what we have with replacement
                    sampled_frames[i] = self.random_state.choice(frames_range, size=self.num_timesteps, replace=True)
                else:
                    # Otherwise sample without replacement
                    sampled_frames[i] = self.random_state.choice(frames_range, size=self.num_timesteps, replace=False)
                # Sort frames to maintain chronological order
                sampled_frames[i].sort()
        else:
            # Even sampling across the delta_frame interval
            frames_indices = np.arange(1, self.num_timesteps + 1)
            # Create frames ensuring we don't exceed dataset bounds
            max_frame = x.shape[0] - 1
            sampled_frames = np.minimum(
                st[:, None] + (delta_frame * frames_indices[None, :]) // self.num_timesteps,
                np.full((st.shape[0], self.num_timesteps), max_frame)
            )

        # Calculate timeframes relative to the start frames
        timeframes = sampled_frames - st[:, None]

        # Extract positions and velocities at the sampled frames
        x_t = x[sampled_frames]  # Shape: (batch_size, num_timesteps, num_particles, num_dimensions)
        v_t = v[sampled_frames]

        # Rearrange axes to match desired shape: (batch_size, num_particles, num_timesteps, num_dimensions)
        x_t = np.transpose(x_t, (0, 2, 1, 3))
        v_t = np.transpose(v_t, (0, 2, 1, 3))

        print('Got {:d} samples!'.format(x_0.shape[0]))

        mole_idx = z
        n_node = mole_idx.shape[0]
        self.n_node = n_node

        _lambda = 1.6

        def d(_i, _j, _t):
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        n = z.shape[0]

        self.Z = torch.Tensor(z)

        atom_edges = torch.zeros(n, n).int()
        for i in range(n):
            for j in range(n):
                if i != j:
                    _d = d(i, j, 0)
                    if _d < _lambda:
                        atom_edges[i][j] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_node):
            for j in range(n_node):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 1])
                        assert not self.atom_edge2[i][j]
                    if self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 2])
                        assert not self.atom_edge[i][j]

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = edges  # [2, edge]

        # pre-compute and store the graph Fourier basis for each graph in the batch
        # use self.edges to compute the graph Fourier basis, same for all graphs across the dataset

        A = to_dense_adj(torch.LongTensor(self.edges), batch=torch.zeros(n_node).long())  # [1, N, N]
        A = A.squeeze(0)  # [N, N] 

        # Compute degree matrix D
        deg = torch.sum(A, dim=1)
        D = torch.diag(deg)

        # Compute Laplacian L = D - A
        L = D - A

        # Compute eigenvalues and eigenvectors, (The eigenvalues are returned in ascending order) 
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        self.U = eigenvectors  # eigenvectors shape: [N, N] 
        
        all_edges = {}

        for i in range(n):
            for j in range(i + 1, n):
                _d = d(i, j, 0)
                if _d < _lambda:
                    idx_i, idx_j = z[i], z[j]
                    if idx_i < idx_j:
                        idx_i, idx_j = idx_j, idx_i
                    if (idx_i, idx_j) in all_edges:
                        all_edges[(idx_i, idx_j)].append([i, j])
                    else:
                        all_edges[(idx_i, idx_j)] = [[i, j]]

        print(all_edges)
        # select the type of bonds to preserve the bond constraint
        conf_edges = []
        for key in all_edges:
            # For MD22, we don't make assumptions about atomic number differences
            conf_edges.extend(all_edges[key])

        print(conf_edges)
        self.conf_edges = conf_edges
        self.x_0, self.v_0, self.x_t, self.v_t = torch.Tensor(x_0), torch.Tensor(v_0), torch.Tensor(x_t), torch.Tensor(
            v_t)
        self.mole_idx = torch.Tensor(mole_idx)
        self.timeframes = torch.Tensor(timeframes) # Shape: (batch_size, num_timesteps) 

        self.cfg = self.sample_cfg()

        print('number of atoms in the molecule:', n_node)

if __name__ == "__main__":
    # load the dataset, and visualize the magnitude of the positions and velocities 
    dataset = MD22DynamicsDataset(partition='train', max_samples=500, delta_frame=3000, molecule_type='AT-AT', data_dir='md22', num_timesteps=8, uneven_sampling=False, internal_seed=None, time_ref=3000)
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