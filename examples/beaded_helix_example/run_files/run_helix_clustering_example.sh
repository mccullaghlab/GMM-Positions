#!/bin/bash
#Set this to your version of lammps+plumed, or remove the plumed part from run script and use a regular lammps compilation
lmp=lmp_plumed
eps=6.0
$lmp -log helix_folding_esp${eps}.log \
           -var outprefix helix_folding_eps$eps \
	   -var plumed_file helix_report.plumed.dat \
           -var steps 50000000  \
           -var eps $eps \
           -in run_helix_plumed.lmp 
