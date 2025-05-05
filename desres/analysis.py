#!/usr/bin/env python3
import MDAnalysis as mda
import argparse
import csv
import os
import py3Dmol
import io

def is_heavy_atom(atom):
    """Check if an atom is a heavy atom (non-hydrogen)"""
    return not atom.name.startswith('H') and not 'H' in atom.name

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze protein trajectory data')
    parser.add_argument('protein_name', type=str, help='Name of the protein (e.g. 1FME)')
    args = parser.parse_args()
    
    # Construct paths based on protein name
    protein_folder = f"desres/{args.protein_name}-0-protein"
    pdb_file = os.path.join(protein_folder, f"{args.protein_name}-0-protein.pdb")
    csv_file = os.path.join(protein_folder, f"{args.protein_name}-0-protein_times.csv")
    
    print(f"Processing protein: {args.protein_name}")
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
    
    print(f"Loaded {len(catalog)} trajectory segments from CSV")
    
    # Extract just the filenames (in the correct order)
    traj_files = [fn for _, fn in catalog]
    
    # Build a single Universe using the protein's PDB file
    u = mda.Universe(pdb_file, *traj_files, dt=200.0)
    
    print(f"Loaded {len(traj_files)} DCD segments → {len(u.trajectory)} total frames")
    print(f"Frame spacing dt = {u.trajectory.dt} ps")
    
    # Filter heavy atoms
    heavy_atoms = [atom for atom in u.atoms if is_heavy_atom(atom)]
    print(f"\nHeavy Atom Information (Total: {len(heavy_atoms)}):")
    for i, atom in enumerate(heavy_atoms):
        print(f"Atom {i+1}: {atom.name} (Type: {atom.type}) - Residue: {atom.resname} {atom.resid}")
    
    # Quick sanity-check: print the first five timesteps with heavy atom coordinates
    print("\nFirst five frames (heavy atoms only):")
    for ts in u.trajectory[:5]:
        print(f"\nFrame {ts.frame:5d} at {ts.time:8.2f} ps:")
        
        # Display only the first 10 heavy atoms to avoid excessive output
        display_count = min(10, len(heavy_atoms))
        for i, atom in enumerate(heavy_atoms[:display_count]):
            coord = atom.position
            print(f"  Atom {atom.id}: {atom.name} (Type: {atom.type}) - Coordinates: ({coord[0]:8.3f}, {coord[1]:8.3f}, {coord[2]:8.3f})")
        
        print(f"  ... and {len(heavy_atoms) - display_count} more heavy atoms")

if __name__ == "__main__":
    main()
