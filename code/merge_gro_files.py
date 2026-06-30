import MDAnalysis as mda
from MDAnalysis import Merge

# Load the two systems
ref = mda.Universe("dna_processed.gro")
mob = mda.Universe("histone_processed.gro")

# Merge the two universes
merged = Merge(ref.atoms, mob.atoms)

# Write to a new GRO file
merged.atoms.write("merged_system.gro")

print("Merged system written to merged_system.gro")
