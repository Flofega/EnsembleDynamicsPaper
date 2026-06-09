#!/bin/bash

#SBATCH --job-name=calcub2
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

for i in $(seq 0 19); do
	cd $i
	gmx_mpi grompp -f md.mdp -c $i.gro -p topol.top -maxwarn 2 -n ../index.ndx
	cd ..
done

mpirun -np 20 gmx_mpi mdrun -plumed plumed.dat -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 -ntomp 1 -nb gpu -pme gpu -bonded gpu 
