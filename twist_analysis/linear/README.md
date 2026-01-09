This file lays out the general plan for extracting twist angles from DNA md simulations. I use three different scripts for this process, they are pretty hard coded so they must be edited for each scenario.

1. Extract data from the simulation using do_x3dna: do_x3dna -f *.xtc -s *.gro -lbpsm -noavg -e * -name _twist

2. run process_twist.py to extract the twist angles from the outputted file (the file used here has twist values for each time step for each base pair step, this script averages the twist values so that there is just one averaged twist angle for each time step)

3. run block_average.py: this block averages the twists over the simulation and outputs averages for four segments of the simulation, then finds the average of that.

4. run plot_twist.py (very much so hardcoded, probably not ideal for analysis of one scenario, it is a comparison plot, block averages)
