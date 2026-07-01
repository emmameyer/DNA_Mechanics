# The breakdown of the contents of this directory follows
1. [initial.sh](initial.sh): runs pdb2gmx, outputs processed gro file, change input/output names per system, step 1
2. [put_it_in_a_box.sh](put_it_in_a_box.sh): puts the processed gro into a cubic box, change input/output names as needed, step 2
3. [first_em.sh](first_em.sh): double precision em (not necessary) done using the box gro file, maxwarn necessary to handle system charges, step 3
4. [solvate_add_ions.sh](solvate_add_ions.sh): solvates the system, adds ions, 0.150 M, runs second em, step 4
5. [nvt_npt.sh](nvt_npt.sh): nvt and npt steps, step 5
6. [correct_trajectory.sh](correct_trajectory.sh): corrects the trajectory md files, change names per usual, ensure EOF is selecting the right group from index file, step 6
7. [sbatch_md_production.slurm](sbatch_md_production.slurm): production run batch script to submit the run as a job on the OSC. It continues from npt, unless altered to continue from a previous production run, step 7.
8. [concatenate.txt](concatenate.txt): gromacs command for combining trajectory files. this is done when extending simulations. this concatenated xtc file is all you need for extending simulations, the gro and tpr file from the second simulation will work for analysis, no need to combine these.

## In general, the step by step process of simulation goes as follows:
1. Obtain PDB
** If multiple strands/objects (such as dsDNA or a DNA and protein system) cause GROMACS errors: Separate the strands/objects to have separate .pdb files (strand1.pdb, strand2.pdb, dna.pdb, histone.pdb, etc.)
* Do this in PyMOL by selecting each object separately and saving them separately by exporting molecule (as a pdb file)


Run gmx pdb2gmx on each object
gmx pdb2gmx -f {object1_name}.pdb -p {object1_name}.top -o {object1_name}_processed.gro -i posre_{object1_name}.itp -water tip3p -ff amber14sb_parmbsc1_cufix -merge all
gmx pdb2gmx -f {object2_name}.pdb -p {object2_name}.top -o {object2_name}_processed.gro -i posre_{object2_name}.itp -water tip3p -ff amber14sb_parmbsc1_cufix -merge all
cp {object*_name}.top {object*_name}.itp
Do this for every object, * replaces numbers for this example
In the new itp files: Delete the include forcefield line at the top and the rest of the lines at the bottom that isn't posre, starting with ; Include water topology down to the [molecules], only thing left at the end should be the position restraint file (take note of the compound names: DNA_chain_A & DNA_chain_B )

cp {one_object}.top system.top
We're using one of the generated objects topology file as a template to make our main system topology file, delete the [moleculetype] and everything below it until water topology, it should include the following:


Include ff
Inlcude "{object_name}.itp"
Include "{object_name}itp"
Water
Posres_water
Ions
System
Molecules: make sure the molecules are correct and in the same order as the include statements (strand1 and strand 2, DNA_chain whatever)

Merge the processed gro files together using MDAnalaysis


import MDAnalysis as mda
# Load the two gromacs files
u1 = mda.Universe("/Users/emma/Documents/kent/minicircles/BSC1/82bp_Tw8_deltaTw0/strand1_processed.gro")
u2 = mda.Universe("/Users/emma/Documents/kent/minicircles/BSC1/82bp_Tw8_deltaTw0/strand2_processed.gro")
# Combine atoms from both universes
combined = mda.Merge(u1.atoms, u2.atoms)
# Save to the same directory
output_path = "/Users/emma/Documents/kent/minicircles/BSC1/82bp_Tw8_deltaTw0/merged.gro"
combined.atoms.write(output_path)


Make the object be in the center of a box aligned to the z-axis
gmx editconf -f system_processed.gro -princ -o system_aligned.gro
gmx editconf -f system_aligned.gro -center 0 0 0 -o system_centered.gro
Here is where you could rotate the system as well using -rotate

Put the system in the center of a box
gmx editconf -f system_centered.gro -o system_box.gro -c -d 1.0 -bt cubic

Then I reserve one hour of interactive time for the following (sinteractive -A PGS0408 -N 1 -n 28 -t 01:00:00):
Run double precision energy minimization on the system prior to solvating (-maxwarn to handle charge)
gmx_d grompp -f inputs/em.mdp -c system_box.gro -p both_strands.top -o em.tpr -maxwarn 1
gmx_d mdrun -v -deffnm em

Solvate the system with a concentration of 0.15 MOL NaCl
gmx solvate -cp system_box.gro -cs spc216.gro -o system_solv.gro -p system.top
gmx grompp -f inputs/ions.mdp -c system_solv.gro -p system.top -o ions.tpr -maxwarn 1
gmx genion -s ions.tpr -o system_solv_ions.gro -p system.top -pname NA -nname CL -neutral -conc 0.150

Second double precision energy minimization:
gmx_d grompp -f inputs/em.mdp -c system_solv_ions.gro -p system.top -o em_2.tpr 
gmx_d mdrun -v -deffnm em_2

NVT and NPT equilibration
gmx grompp -f inputs/nvt.mdp -c em_2.gro -r em_2.gro -p system.top -o nvt.tpr
gmx mdrun -v -deffnm nvt
gmx grompp -f inputs/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p system.top -o npt.tpr
gmx mdrun -v -deffnm npt

Production run (short, 0.5 ns, runs can be done in the hour of reserved time, other wise, must submit as a job using gpu)
gmx grompp -f inputs/md.mdp -c npt.gro -t npt.cpt -p system.top -o md.tpr
gmx mdrun -deffnm md
