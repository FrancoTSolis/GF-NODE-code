import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
import os

def compute_bond_lengths(positions, bond_pairs):
    """
    Compute bond lengths given positions and bond pairs.

    Parameters:
    - positions: numpy array of shape (n_samples, n_atoms, 3)
    - bond_pairs: list of tuples, each tuple contains indices of bonded atoms (atom_i, atom_j)

    Returns:
    - bond_lengths: numpy array of shape (n_samples, n_bonds)
    """
    bond_lengths = []
    for atom_i, atom_j in bond_pairs:
        vec = positions[:, atom_i, :] - positions[:, atom_j, :]
        length = np.linalg.norm(vec, axis=1)
        bond_lengths.append(length)
    bond_lengths = np.array(bond_lengths).T  # Shape: (n_samples, n_bonds)
    return bond_lengths

def compute_bond_angles(positions, angle_triplets):
    """
    Compute bond angles given positions and angle triplets.

    Parameters:
    - positions: numpy array of shape (n_samples, n_atoms, 3)
    - angle_triplets: list of tuples, each tuple contains indices of three atoms (atom_i, atom_j, atom_k)

    Returns:
    - bond_angles: numpy array of shape (n_samples, n_angles)
    """
    bond_angles = []
    for atom_i, atom_j, atom_k in angle_triplets:
        vec_ij = positions[:, atom_i, :] - positions[:, atom_j, :]
        vec_kj = positions[:, atom_k, :] - positions[:, atom_j, :]
        # Normalize vectors
        vec_ij_norm = vec_ij / np.linalg.norm(vec_ij, axis=1)[:, None]
        vec_kj_norm = vec_kj / np.linalg.norm(vec_kj, axis=1)[:, None]
        # Compute dot product and angle
        dot_product = np.sum(vec_ij_norm * vec_kj_norm, axis=1)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # In radians
        bond_angles.append(np.degrees(angle))  # Convert to degrees if desired
    bond_angles = np.array(bond_angles).T  # Shape: (n_samples, n_angles)
    return bond_angles

def compute_errors(predicted, ground_truth, relative=False):
    """
    Compute MAE and RMSE between predicted and ground truth values.

    Parameters:
    - predicted: numpy array of shape (n_samples, n_properties)
    - ground_truth: numpy array of shape (n_samples, n_properties)
    - relative: bool, if True compute relative errors

    Returns:
    - mae: Mean Absolute Error
    - rmse: Root Mean Square Error
    """
    error = predicted - ground_truth
    if relative:
        error = error / ground_truth
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    return mae, rmse

def compute_dihedral_angles(positions, torsion_quartets):
    """
    Compute dihedral angles given positions and torsion quartets.

    Parameters:
    - positions: numpy array of shape (n_samples, n_atoms, 3)
    - torsion_quartets: list of tuples, each tuple contains indices of four atoms (atom_i, atom_j, atom_k, atom_l)

    Returns:
    - dihedral_angles: numpy array of shape (n_samples, n_torsions)
    """
    dihedral_angles = []
    for atom_i, atom_j, atom_k, atom_l in torsion_quartets:
        p0 = positions[:, atom_i, :]
        p1 = positions[:, atom_j, :]
        p2 = positions[:, atom_k, :]
        p3 = positions[:, atom_l, :]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize b1 so that it does not influence magnitude of vector rejections
        b1 /= np.linalg.norm(b1, axis=1)[:, None]

        # Compute the vector rejections
        v = b0 - (np.sum(b0 * b1, axis=1)[:, None]) * b1
        w = b2 - (np.sum(b2 * b1, axis=1)[:, None]) * b1

        # Compute angle between v and w
        x = np.sum(v * w, axis=1)
        y = np.sum(np.cross(b1, v) * w, axis=1)
        angle = np.arctan2(y, x)
        dihedral_angles.append(np.degrees(angle))  # Convert to degrees
    dihedral_angles = np.array(dihedral_angles).T  # Shape: (n_samples, n_torsions)
    return dihedral_angles

def compute_kde_and_distribution_shift(phi_gt, psi_gt, phi_pred, psi_pred, save_path):
    """
    Compute the KDE of the Ramachandran plots and calculate the distribution shift
    between ground truth and predicted distributions.

    Parameters:
    - phi_gt, psi_gt: Lists or arrays of ground truth phi and psi angles
    - phi_pred, psi_pred: Lists or arrays of predicted phi and psi angles
    - save_path: Path to save the plots and results
    """
    # Prepare data
    gt_angles = np.vstack([phi_gt, psi_gt])
    pred_angles = np.vstack([phi_pred, psi_pred])

    # Compute KDEs
    gt_kde = gaussian_kde(gt_angles)
    pred_kde = gaussian_kde(pred_angles)

    # Evaluate KDEs on a grid
    x_grid, y_grid = np.mgrid[-180:180:100j, -180:180:100j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    gt_kde_vals = np.reshape(gt_kde(grid_coords), x_grid.shape)
    pred_kde_vals = np.reshape(pred_kde(grid_coords), x_grid.shape)

    # Normalize KDE values
    gt_kde_vals /= np.sum(gt_kde_vals)
    pred_kde_vals /= np.sum(pred_kde_vals)

    # Compute KL divergence
    kl_divergence = entropy(pred_kde_vals.ravel(), gt_kde_vals.ravel())
    print(f"KL Divergence between predicted and ground truth Ramachandran distributions: {kl_divergence}")

    # Plot KDEs
    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(gt_kde_vals), cmap=plt.cm.gist_earth_r,
               extent=[-180, 180, -180, 180])
    plt.xlabel('Phi (ϕ) Angle (degrees)')
    plt.ylabel('Psi (ψ) Angle (degrees)')
    plt.title('Ground Truth Ramachandran KDE')
    plt.savefig(os.path.join(save_path, 'ramachandran_kde_gt.pdf'))
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(pred_kde_vals), cmap=plt.cm.gist_earth_r,
               extent=[-180, 180, -180, 180])
    plt.xlabel('Phi (ϕ) Angle (degrees)')
    plt.ylabel('Psi (ψ) Angle (degrees)')
    plt.title('Predicted Ramachandran KDE')
    plt.savefig(os.path.join(save_path, 'ramachandran_kde_pred.pdf'))
    plt.close()

def plot_ramachandran(phi_angles, psi_angles, title='Ramachandran Plot', save_path=None):
    """
    Generate a Ramachandran plot of phi and psi torsion angles.

    Parameters:
    - phi_angles: array of phi angles
    - psi_angles: array of psi angles
    - title: title of the plot
    - save_path: path to save the plot
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(phi_angles % 360 - 180, psi_angles % 360 - 180, s=1, alpha=0.5)
    plt.xlabel('Phi (ϕ) Angle (degrees)')
    plt.ylabel('Psi (ψ) Angle (degrees)')
    plt.title(title)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xticks(range(-180, 181, 60))
    plt.yticks(range(-180, 181, 60))
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



