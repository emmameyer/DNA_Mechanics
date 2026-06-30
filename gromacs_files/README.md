1. initial.sh: runs pdb2gmx, outputs processed gro file, change input/output names per system, step 1
2. put_it_in_a_box.sh: puts the processed gro into a cubic box, change input/output names as needed, step 2
3. first_em.sh: double precision em (not necessary) done using the box gro file, maxwarn necessary to handle system charges, step 3
4. 
