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


args = parser.parse_args()
if args.config_by_file:
    job_param_path = 'configs/' + args.config
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
args.time = time.strftime("%m%d%H%M%S")
if args.no_fourier: 
    args.exp_name = args.exp_name + "_" + args.mol + "_" + "no_fourier" + "_" + args.time
elif args.no_ode: 
    args.exp_name = args.exp_name + "_" + args.mol + "_" + args.fourier_basis + "_" + "no_ode" + "_" + args.time
else: 
    args.exp_name = args.exp_name + "_" + args.mol + "_" + args.fourier_basis + "_" + args.time

if not args.no_fourier: 
    assert args.fourier_basis in ['linear', 'graph'], "fourier_basis must be either 'linear' or 'graph'"

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass


# save the config file with pretty print 
json_object = json.dumps(vars(args), indent=4)
with open(args.outf + "/" + args.exp_name + "/config.json", "w") as outfile:
    outfile.write(json_object)

# Open the file with no buffering
file_descriptor = os.open(args.outf + "/" + args.exp_name + "/train.log", 
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
    else:
        raise Exception("Wrong model specified")

    print(model)

    model_save_path = args.outf + '/' + args.exp_name + '/' + 'saved_model.pth'
    print(f'Model saved to {model_save_path}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=2500, gamma=0.5)

    results = {'epochs': [], 'loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)
                print("==> best model saved at epoch %d" % epoch)
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best apoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))
        scheduler.step()

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)
    return best_train_loss, best_val_loss, best_test_loss, best_epoch


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


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)


# Close the file descriptor at the end of the program
os.close(file_descriptor)