1. [initial.sh](initial.sh): runs pdb2gmx, outputs processed gro file, change input/output names per system, step 1
2. [put_it_in_a_box.sh](put_it_in_a_box.sh): puts the processed gro into a cubic box, change input/output names as needed, step 2
3. [first_em.sh](first_em.sh): double precision em (not necessary) done using the box gro file, maxwarn necessary to handle system charges, step 3
4. [solvate_add_ions.sh](solvate_add_ions.sh): solvates the system, adds ions, 0.150 M, runs second em, step 4
5. [nvt_npt.sh](nvt_npt.sh): nvt and npt steps, step 5
6. [correct_trajectory.sh](correct_trajectory.sh): corrects the trajectory md files, change names per usual, ensure EOF is selecting the right group from index file, step 6
7. [sbatch_md_production.slurm](sbatch_md_production.slurm): production run batch script to submit the run as a job on the OSC. It continues from npt, unless altered to continue from a previous production run, step 7.
