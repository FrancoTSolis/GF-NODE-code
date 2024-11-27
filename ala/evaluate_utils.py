import numpy as np
import matplotlib.pyplot as plt

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

def compute_errors(predicted, ground_truth):
    """
    Compute MAE and RMSE between predicted and ground truth values.

    Parameters:
    - predicted: numpy array of shape (n_samples, n_properties)
    - ground_truth: numpy array of shape (n_samples, n_properties)

    Returns:
    - mae: Mean Absolute Error
    - rmse: Root Mean Square Error
    """
    error = predicted - ground_truth
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



def plot_ramachandran(phi_angles, psi_angles, title='Ramachandran Plot', save_path=None):
    """
    Generate a Ramachandran plot of phi and psi torsion angles.

    Parameters:
    - phi_angles: array of phi angles
    - psi_angles: array of psi angles
    - title: title of the plot
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
    # plt.show()
    # save plot pdf
    plt.savefig(save_path)



