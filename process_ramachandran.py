import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
import os

def plot_ramachandran_comparison(model_files, save_path=None):
    """
    Plots Ramachandran plots from multiple models side by side for comparison.

    Parameters:
    - model_files: list of dictionaries with keys 'name', 'pred_path', 'gt_path'
    - save_path: path to save the combined plot
    """
    num_models = len(model_files)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 8))

    if num_models == 1:
        axes = [axes]

    for ax, model_info in zip(axes, model_files):
        model_name = model_info['name']
        pred_file_path = model_info['pred_path']
        gt_file_path = model_info['gt_path']

        data_pred = np.load(pred_file_path)
        data_gt = np.load(gt_file_path)

        phi_pred = data_pred['phi'] % 360 - 180
        psi_pred = data_pred['psi'] % 360 - 180
        phi_gt = data_gt['phi'] % 360 - 180
        psi_gt = data_gt['psi'] % 360 - 180

        # Compute KDEs
        values_pred = np.vstack([phi_pred, psi_pred])
        values_gt = np.vstack([phi_gt, psi_gt])

        kde_pred = gaussian_kde(values_pred)
        kde_gt = gaussian_kde(values_gt)

        X, Y = np.mgrid[-180:180:100j, -180:180:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z_pred = np.reshape(kde_pred(positions), X.shape)
        Z_gt = np.reshape(kde_gt(positions), X.shape)

        # Compute distribution shift (KL divergence)
        Z_pred_flat = Z_pred.ravel()
        Z_gt_flat = Z_gt.ravel()
        Z_pred_flat /= np.sum(Z_pred_flat)
        Z_gt_flat /= np.sum(Z_gt_flat)
        kl_divergence = entropy(Z_pred_flat, Z_gt_flat)

        # Plot KDE
        ax.imshow(np.rot90(Z_pred), cmap=plt.cm.gist_earth_r,
                  extent=[-180, 180, -180, 180])
        ax.set_xlabel('Phi (ϕ) Angle (degrees)')
        ax.set_ylabel('Psi (ψ) Angle (degrees)')
        ax.set_title(f'{model_name}\nKL Divergence: {kl_divergence:.4f}')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-180, 181, 60))
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage: update with actual model names and file paths
    model_files = [
        {
            'name': 'Model A',
            'pred_path': 'model_a_output/torsion_angles_pred.npz',
            'gt_path': 'model_a_output/torsion_angles_gt.npz'
        },
        {
            'name': 'Model B',
            'pred_path': 'model_b_output/torsion_angles_pred.npz',
            'gt_path': 'model_b_output/torsion_angles_gt.npz'
        },
        {
            'name': 'Ground Truth',
            'pred_path': 'ground_truth_output/torsion_angles_gt.npz',
            'gt_path': 'ground_truth_output/torsion_angles_gt.npz'
        }
    ]
    plot_ramachandran_comparison(model_files, save_path='ramachandran_comparison.pdf') 