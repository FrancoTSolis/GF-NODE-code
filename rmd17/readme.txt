THE REVISED MD17 dataset:
=========================


Citation:
========

    Anders S. Christensen and O. Anatole von Lilienfeld (2020) "On the role of gradients for machine learning of molecular energies and forces" arXiv:?????

The molecules are taken from the original MD17 dataset by Chmiela et al., and 100,000 structures are taken, and the energies and forces are recalculated at the PBE/def2-SVP level of theory using very tight SCF convergence and very dense DFT integration grid. As such, the dataset is practically free from nummerical noise. 


One warning: As the structures are taken from a molecular dynamics simulation (i.e. time series data), they are not guaranteed to be independent samples. This is easily evident from the autocorrelation function for the original MD17 dataset

In short: DO NOT train a model on more than 1000 samples from this dataset. Data already published with 50K samples on the original MD17 dataset should be considered meaningless due to this fact and due to the noise in the original data.


The data:
=========

The ten molecules are save in Numpy .npz format.

The keys correspond to:

'nuclear_charges'   : The nuclear charges for the molecule
'coords'            : The coordinates for each conformation (in units of ångstrom)
'energies'          : The total energy of each conformation (in units of kcal/mol)
'forces'            : The cartesian forces of each conformation (in units of kcal/mol/ångstrom)
'old_indices'       : The index of each conformation in the original MD17 dataset
'old_energies'      : The energy of each conformation taken from the original MD17 dataset  (in units of kcal/mol)
'old_forces'        : The forces of each conformation taken from the original MD17 dataset  (in units of kcal/mol/ångstrom)

*Note that for Azobenzene, only 99988 samples are available due to 11 failed DFT calculations due to van der Walls clash, and the original dataset only contained 99999 structures.


Data splits:
============
Five training and test splits are saved in CSV format containing the corresponding indices.

