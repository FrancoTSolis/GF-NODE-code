import argparse
from argparse import Namespace
import torch
import torch.utils.data
# from md17.dataset import MD17DynamicsDataset as MD17Dataset
from model.fourier_md import FourierMD 
import os, sys, time 
from torch import nn, optim
import json
from torch.optim.lr_scheduler import StepLR
from ala.evaluate_utils import compute_bond_lengths, compute_bond_angles, compute_dihedral_angles, compute_errors, plot_ramachandran, compute_kde_and_distribution_shift
import matplotlib.pyplot as plt 

import random
import numpy as np

parser = argparse.ArgumentParser(description='FourierMD')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_timesteps', type=int, default=8, metavar='N',
                    help='number of time steps per sample')
parser.add_argument('--use_time_conv', type=eval, default=False)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='log/md17', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='FourierMD', metavar='N',
                    help='available models: FourierMD')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=5, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='weight decay')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--delta_frame', type=int, default=50,
                    help='Number of frames delta.')
parser.add_argument('--mol', type=str, default='aspirin',
                    help='Name of the molecule.')
parser.add_argument('--data_dir', type=str, default='',
                    help='Data directory.')
parser.add_argument('--learnable', type=eval, default=False, metavar='N',
                    help='Use learnable FK.')

parser.add_argument("--config_by_file", default=False, action="store_true", )
parser.add_argument("--config", default='config_md17_no.json', 
                    type=str, help='Path to the config file.')

# ODE solver parameters 
parser.add_argument('--solver', type=str, help='ODE solver') 
parser.add_argument('--rtol', type=float, help='Relative tolerance for ODE solver')
parser.add_argument('--atol', type=float, help='Absolute tolerance for ODE solver')

parser.add_argument('--num_modes', type=int, default=2, help='The number of modes.')
parser.add_argument('--time_emb_dim', type=int, default=32,
                    help='The dimension of time embedding.')
parser.add_argument('--norm', action='store_true', default=False,
                    help='Use norm in FourierMD')
parser.add_argument('--flat', action='store_true', default=False,
                    help='flat MLP')

# uneven sampling 
parser.add_argument('--uneven_sampling', action='store_true', default=False,
                    help='Use uneven sampling')
parser.add_argument('--internal_seed', type=int, default=0,
                    help='Internal seed for uneven sampling')

parser.add_argument('--fourier_basis', type=str, default=None, 
                    help='Fourier basis for time convolution, either linear or graph') 

# ablation study 
parser.add_argument('--no_ode', action='store_true', default=False,
                    help='No ODE block')
parser.add_argument('--no_fourier', action='store_true', default=False,
                    help='No Fourier block')


parser.add_argument('--ref_path', type=str, default='log/exp_1', metavar='N',
                    help='folder from which to load the config file')


args = parser.parse_args()

job_param_path = os.path.join(args.ref_path, 'config.json')
with open(job_param_path, 'r') as f:
    hyper_params = json.load(f)
    # update keys existing in config
    args = vars(args)
    args.update((k, v) for k, v in hyper_params.items() if k in args)
    args = Namespace(**args)

# assert torch.cuda.is_available(), "no cuda device available"
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.mol == "ala2": 
    from ala.dataset import AlanineDataset as MoleculeDynamicsDataset 
elif args.mol in ["aspirin", "benzene_old", "ethanol", "malonaldehyde", "naphthalene", "salicylic", "toluene", "uracil"]: 
    from md17.dataset import MD17DynamicsDataset as MoleculeDynamicsDataset
else: 
    raise ValueError(f"Molecule {args.mol} not supported") 
 

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss(reduction='none')

assert not (args.no_fourier and args.no_ode), "Cannot remove both Fourier and ODE block" 
# exp_name plus a time stamp in format MMDDHHMMSS 

if not args.no_fourier: 
    assert args.fourier_basis in ['linear', 'graph'], "fourier_basis must be either 'linear' or 'graph'"


# Open the file with no buffering
file_descriptor = os.open(os.path.join(args.ref_path, 'evaluate.log'),
                          os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

# Duplicate the file descriptor to stdout (fd 1)
os.dup2(file_descriptor, sys.stdout.fileno())

print(args)
# torch.autograd.set_detect_anomaly(True)


def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train = MoleculeDynamicsDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                molecule_type=args.mol, delta_frame=args.delta_frame,
                                num_timesteps=args.num_timesteps, 
                                uneven_sampling=args.uneven_sampling, internal_seed=args.internal_seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0)

    dataset_val = MoleculeDynamicsDataset(partition='val', max_samples=2000, data_dir=args.data_dir,
                                molecule_type=args.mol, delta_frame=args.delta_frame,
                                num_timesteps=args.num_timesteps, 
                                uneven_sampling=args.uneven_sampling, internal_seed=args.internal_seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=0)

    dataset_test = MoleculeDynamicsDataset(partition='test', max_samples=2000, data_dir=args.data_dir,
                                molecule_type=args.mol, delta_frame=args.delta_frame,
                                num_timesteps=args.num_timesteps, 
                                uneven_sampling=args.uneven_sampling, internal_seed=args.internal_seed)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=0)
    
    delta_frame = args.delta_frame

    if args.no_fourier:
        assert dataset_train.n_node == args.num_modes, "Number of modes must be the same as the number of atoms" 
        assert args.fourier_basis == None, "No Fourier block, so fourier_basis must be None" 

    if args.model == 'fourier': 
        model = FourierMD(n_layers=args.n_layers, in_node_nf=2, in_edge_nf=2 + 3, hidden_nf=args.nf, device=device,
                          with_v=True, flat=args.flat, activation=nn.SiLU(), norm=args.norm, num_modes=args.num_modes,
                          num_timesteps=args.num_timesteps, time_emb_dim=args.time_emb_dim, 
                          num_atoms=None, solver=args.solver, rtol=args.rtol, atol=args.atol, 
                          delta_frame=delta_frame, fourier_basis=args.fourier_basis, 
                          no_ode=args.no_ode, no_fourier=args.no_fourier)
        
        # load the model weights ")
        model_save_path = os.path.join(args.ref_path, 'saved_model.pth') 
        print(f'Loading model from {model_save_path}') 
        model.load_state_dict(torch.load(model_save_path)) 
        # print("provisional not loading the model weights")

    else:
        raise Exception("Wrong model specified")

    print(model)

    with torch.no_grad():
        evaluate(model, loader_test) 


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        if model.num_atoms is None: 
            model.num_atoms = n_nodes
        else: 
            assert model.num_atoms == n_nodes, "Number of atoms should be the same" 
        data, cfg = data[:-1], data[-1]
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(-1)) for d in data]  # construct mini-batch graphs
        for i in [4, 5]:
            d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
            data[i] = d.transpose(0, 1).contiguous().view(-1, 3)
        loc, vel, edge_attr, charges, loc_end, vel_end, Z, timeframes, U_batch = data

        U_batch = U_batch.view(batch_size, n_nodes, n_nodes)

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        cfg = loader.dataset.get_cfg(batch_size, n_nodes, cfg)
        cfg = {_: cfg[_].to(device) for _ in cfg}

        optimizer.zero_grad()

        if args.model == 'fourier':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_mean = loc.view(batch_size, n_nodes, 3).mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, 3)  # [BN, 3]
            loc_pred, vel_pred, _ = model(loc.detach(), nodes, edges, edge_attr, vel, loc_mean=loc_mean, timeframes=timeframes, 
                                          U_batch=U_batch) 
        else:
            raise Exception("Wrong model")

        losses = loss_mse(loc_pred, loc_end).view(args.num_timesteps, batch_size * n_nodes, 3)
        losses = torch.mean(losses, dim=(1, 2))
        loss = torch.mean(losses)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += losses[-1].item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']


def evaluate(model, loader):

    # Non-hydrogen atom indices (NumPy indices)
    atom_indices = {
        'CH3_ACE': 1,    # CH3 in ACE
        'C_ACE': 4,      # C in ACE
        'O_ACE': 5,      # O in ACE
        'N_ALA': 6,      # N in ALA
        'CA_ALA': 8,     # CA in ALA
        'CB_ALA': 10,    # CB in ALA
        'C_ALA': 14,     # C in ALA
        'O_ALA': 15,     # O in ALA
        'N_NME': 16,     # N in NME
        'CH3_NME': 18    # CH3 in NME
    }
    
    # Bond pairs (non-hydrogen atoms)
    bond_pairs = [
        (atom_indices['CH3_ACE'], atom_indices['C_ACE']),   # CH3 (ACE) - C (ACE)
        (atom_indices['C_ACE'], atom_indices['O_ACE']),     # C (ACE) - O (ACE)
        (atom_indices['C_ACE'], atom_indices['N_ALA']),     # C (ACE) - N (ALA)
        (atom_indices['N_ALA'], atom_indices['CA_ALA']),    # N (ALA) - CA (ALA)
        (atom_indices['CA_ALA'], atom_indices['CB_ALA']),   # CA (ALA) - CB (ALA)
        (atom_indices['CA_ALA'], atom_indices['C_ALA']),    # CA (ALA) - C (ALA)
        (atom_indices['C_ALA'], atom_indices['O_ALA']),     # C (ALA) - O (ALA)
        (atom_indices['C_ALA'], atom_indices['N_NME']),     # C (ALA) - N (NME)
        (atom_indices['N_NME'], atom_indices['CH3_NME'])    # N (NME) - CH3 (NME)
    ]
    
    # Angle triplets (non-hydrogen atoms)
    angle_triplets = [
        (atom_indices['CH3_ACE'], atom_indices['C_ACE'], atom_indices['O_ACE']),
        (atom_indices['CH3_ACE'], atom_indices['C_ACE'], atom_indices['N_ALA']),
        (atom_indices['O_ACE'], atom_indices['C_ACE'], atom_indices['N_ALA']),
        (atom_indices['C_ACE'], atom_indices['N_ALA'], atom_indices['CA_ALA']),
        (atom_indices['N_ALA'], atom_indices['CA_ALA'], atom_indices['CB_ALA']),
        (atom_indices['N_ALA'], atom_indices['CA_ALA'], atom_indices['C_ALA']),
        (atom_indices['CB_ALA'], atom_indices['CA_ALA'], atom_indices['C_ALA']),
        (atom_indices['CA_ALA'], atom_indices['C_ALA'], atom_indices['O_ALA']),
        (atom_indices['CA_ALA'], atom_indices['C_ALA'], atom_indices['N_NME']),
        (atom_indices['O_ALA'], atom_indices['C_ALA'], atom_indices['N_NME']),
        (atom_indices['C_ALA'], atom_indices['N_NME'], atom_indices['CH3_NME'])
    ]
    
    # Torsion quartets for phi and psi angles
    torsion_quartets = [
        # Phi angle: C (ACE) - N (ALA) - CA (ALA) - C (ALA)
        (atom_indices['C_ACE'], atom_indices['N_ALA'], atom_indices['CA_ALA'], atom_indices['C_ALA']),
        
        # Psi angle: N (ALA) - CA (ALA) - C (ALA) - N (NME)
        (atom_indices['N_ALA'], atom_indices['CA_ALA'], atom_indices['C_ALA'], atom_indices['N_NME'])
    ]

    model.eval()
    bond_length_errors = []
    bond_angle_errors = []
    bond_length_rel_errors = []
    bond_angle_rel_errors = []

    phi_angles_pred = []
    psi_angles_pred = []
    phi_angles_gt = []
    psi_angles_gt = []

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        if model.num_atoms is None: 
            model.num_atoms = n_nodes
        else: 
            assert model.num_atoms == n_nodes, "Number of atoms should be the same" 
        data, cfg = data[:-1], data[-1]
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(-1)) for d in data]  # construct mini-batch graphs
        for i in [4, 5]:
            d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
            data[i] = d.transpose(0, 1).contiguous().view(-1, 3)
        loc, vel, edge_attr, charges, loc_end, vel_end, Z, timeframes, U_batch = data

        U_batch = U_batch.view(batch_size, n_nodes, n_nodes)

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        cfg = loader.dataset.get_cfg(batch_size, n_nodes, cfg)
        cfg = {_: cfg[_].to(device) for _ in cfg}


        if args.model == 'fourier':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_mean = loc.view(batch_size, n_nodes, 3).mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, 3)  # [BN, 3]
            loc_pred, vel_pred, _ = model(loc.detach(), nodes, edges, edge_attr, vel, loc_mean=loc_mean, timeframes=timeframes, 
                                          U_batch=U_batch) 
        else:
            raise Exception("Wrong model")
        
        # Convert to numpy
        loc_pred_np = loc_pred.cpu().numpy().reshape(-1, n_nodes, 3)
        loc_gt_np = loc_end.cpu().numpy().reshape(-1, n_nodes, 3)        

        # Compute bond lengths and angles
        bond_lengths_pred = compute_bond_lengths(loc_pred_np, bond_pairs)
        bond_lengths_gt = compute_bond_lengths(loc_gt_np, bond_pairs)

        # Absolute errors
        mae_bond_length, _ = compute_errors(bond_lengths_pred, bond_lengths_gt)
        bond_length_errors.append(mae_bond_length)

        # Relative errors
        bond_length_rel_error = np.abs((bond_lengths_pred - bond_lengths_gt) / bond_lengths_gt)
        mae_bond_length_rel = np.mean(bond_length_rel_error)
        bond_length_rel_errors.append(mae_bond_length_rel)

        bond_angles_pred = compute_bond_angles(loc_pred_np, angle_triplets)
        bond_angles_gt = compute_bond_angles(loc_gt_np, angle_triplets)

        # Absolute errors
        mae_bond_angle, _ = compute_errors(bond_angles_pred, bond_angles_gt)
        bond_angle_errors.append(mae_bond_angle)

        # Relative errors
        bond_angle_rel_error = np.abs((bond_angles_pred - bond_angles_gt) / bond_angles_gt)
        mae_bond_angle_rel = np.mean(bond_angle_rel_error)
        bond_angle_rel_errors.append(mae_bond_angle_rel)

        # Compute torsion angles
        dihedral_angles_pred = compute_dihedral_angles(loc_pred_np, torsion_quartets)
        dihedral_angles_gt = compute_dihedral_angles(loc_gt_np, torsion_quartets)

        phi_angles_pred.extend(dihedral_angles_pred[:, 0])
        psi_angles_pred.extend(dihedral_angles_pred[:, 1])
        phi_angles_gt.extend(dihedral_angles_gt[:, 0])
        psi_angles_gt.extend(dihedral_angles_gt[:, 1])

    # Aggregate results
    avg_bond_length_error = np.mean(bond_length_errors)
    avg_bond_angle_error = np.mean(bond_angle_errors)
    avg_bond_length_rel_error = np.mean(bond_length_rel_errors)
    avg_bond_angle_rel_error = np.mean(bond_angle_rel_errors)

    print(f"Average Bond Length MAE: {avg_bond_length_error}")
    print(f"Average Bond Length Relative MAE: {avg_bond_length_rel_error}")
    print(f"Average Bond Angle MAE: {avg_bond_angle_error}")
    print(f"Average Bond Angle Relative MAE: {avg_bond_angle_rel_error}")
    
    # Plot Ramachandran Plot
    plot_ramachandran(np.array(phi_angles_gt), np.array(psi_angles_gt), title='Ground Truth Ramachandran Plot', save_path=os.path.join(args.ref_path, 'ramachandran_gt.pdf'))
    plot_ramachandran(np.array(phi_angles_pred), np.array(psi_angles_pred), title='Predicted Ramachandran Plot', save_path=os.path.join(args.ref_path, 'ramachandran_pred.pdf'))
    
    # Overlay plots for comparison
    plt.figure(figsize=(8, 8))
    plt.scatter(np.array(phi_angles_gt) % 360 - 180, np.array(psi_angles_gt) % 360 - 180, s=1, alpha=0.5, label='Ground Truth')
    plt.scatter(np.array(phi_angles_pred) % 360 - 180, np.array(psi_angles_pred) % 360 - 180, s=1, alpha=0.5, label='Predicted')
    plt.xlabel('Phi (ϕ) Angle (degrees)')
    plt.ylabel('Psi (ψ) Angle (degrees)')
    plt.title('Ramachandran Plot Comparison')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xticks(range(-180, 181, 60))
    plt.yticks(range(-180, 181, 60))
    plt.grid(True)
    plt.legend()
    # plt.show()
    # plt.savefig('ramachandran_plot_comparison.pdf') 
    plt.savefig(os.path.join(args.ref_path, 'ramachandran_plot_comparison.pdf'))



    # Compute KDE and distribution shift
    compute_kde_and_distribution_shift(
        phi_angles_gt, psi_angles_gt, phi_angles_pred, psi_angles_pred, args.ref_path
    )

    # Save torsion angles to files for later processing
    np.savez(os.path.join(args.ref_path, 'torsion_angles_pred.npz'),
             phi=phi_angles_pred, psi=psi_angles_pred)
    np.savez(os.path.join(args.ref_path, 'torsion_angles_gt.npz'),
             phi=phi_angles_gt, psi=psi_angles_gt)


if __name__ == "__main__":
    main()


# Close the file descriptor at the end of the program
os.close(file_descriptor)