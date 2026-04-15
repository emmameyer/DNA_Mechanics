README file for plotters
1. fig_style.py: script to ensure same format across all figures
2. plot_E_circ.py: plots the free energies of circularization considering 1) twist energy, 2) bending energy, and 3) fluctuations about a closed circle (following some mathematics of Shimada Yamakawa paper).
3. plot_exp_vs_TBC_h_0.py: intended to compare the experimentally determined relaxed helical repeat h_0 according to split band data of Soumya to the Marko Siggia prediction using the data available in 1994 (G/C ~ 2.4). This is done by performing the same curve-fitting done by MS to the experimental split band locations (via computing the helical repeat N_bp/Tw, where Tw is approximately a half-integer at the splits). Produces a plot with curves for: no additives, two EtBr, HMfB, and two temperature cases.
4. plot_h.py: plot helical repeat (h) over time for one or more simulations (simple minicircular DNA or minicircular DNA with proteins)
5. plot_hbond.py: plots hydrogen bond analysis (broken/intact interactions over the length of the simulation)
6. plot_stacking: plots stacking interaction analysis (broken/intact interactions over the length of the simulation)
7. plot_sum_of_broken_interactions.py: finds the total number of broken hydrogen bonds and the total number of broken stacking interactions for a circular dsDNA trajectory, and plots the two values as a function of time.
8. plot_sawtooth_v1:
9. plot_sawtooth_NA_before:
10. plot_sawtooth_v2:
