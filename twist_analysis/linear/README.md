This file lays out the general plan for extracting twist angles from DNA md simulations. I use three different scripts for this process, they are pretty hard coded so they must be edited for each scenario.

1. Extract data from the simulation using do_x3dna: do_x3dna -f *.xtc -s *.gro -lbpsm -noavg -e {number of last frame} -name _twist

2. run process_twist.py to extract the twist angles from the outputted file (the file included roll, tilt, ... etc. not important to us, so we only want the twist values).

3. run block_average.py: this block averages the twists over the simulation and outputs averages for four segments of the simulation, then finds the average of that.

4. run plot_twist.py (very much so hardcoded, probably not ideal one one scenario, it is a comparison plot, block averages)