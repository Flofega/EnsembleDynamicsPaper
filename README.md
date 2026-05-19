# EnsemblePaper
Files accompanying the publication: 

# Contents

Each folder contains the files necessary to train the models used in the publication and also all the files to perform any of the enhanced sampling simulations. The training scripts all require data from a short biasing simulation.
The Python scripts require ``MDAnalysis`` and a fork of ``mlcolvar`` ([1.0.0+1218.g67d6d6b](https://github.com/DevergneTimothee/mlcolvar)).

The enhanced sampling simulations were performed using Plumed 2.10, gromacs 2024.4, lammps 22 July 2025, CP2K 2024.3.
The plumed inputs require addtional functionalities developed for this publications. All the extra functions and modified source files can be found in the folder `plumed_so` and are ready to be compiled with `plumed mklib`.
A Plumed version with all the functionalities integrated can be found at: https://github.com/Flofega/EnsembleDynamics.

