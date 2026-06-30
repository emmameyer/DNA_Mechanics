"""
make_wc_pairs_ndx.py

Builds a GROMACS index file with Watson-Crick H-bond groups for a set of
base pairs. Two flavors get written: H-D-A triplets (donor H, donor heavy
atom, acceptor) for gmx angle, and H-A pairs for gmx distance.

Solvent and ions are skipped automatically since we only look at DA/DT/DG/DC
residues. The numbers in pairs.tsv refer to positions within that DNA-only
list (1..N), not the raw .gro residue numbers.

Pairing rules:
  G donor: N1-H1, pairs with C acceptor N3
  T donor: N3-H3, pairs with A acceptor N1

Usage:
  python make_wc_pairs_ndx.py structure.gro pairs.tsv output.ndx

pairs.tsv is just two columns, one pair per line, e.g. for a 43 bp duplex:
  1 86
  2 85
  ...
  43 44
"""
from pathlib import Path
import sys


def normalize_resname(rn):
    # strip digits/whitespace, keep letters, uppercase, then match known
    # prefixes so things like DA5, dt3, ADE all collapse to one tag
    letters = ''.join(ch for ch in rn if ch.isalpha()).upper()
    for prefix in ("DA", "DT", "DG", "DC", "ADE", "THY", "GUA", "CYT", "A", "T", "G", "C"):
        if letters.startswith(prefix):
            return prefix
    return letters


def base_letter_from_resname(rn_norm):
    if rn_norm.startswith("DA") or rn_norm in ("ADE", "A"):
        return "A"
    if rn_norm.startswith("DT") or rn_norm in ("THY", "T"):
        return "T"
    if rn_norm.startswith("DG") or rn_norm in ("GUA", "G"):
        return "G"
    if rn_norm.startswith("DC") or rn_norm in ("CYT", "C"):
        return "C"
    return None


def parse_gro_atoms(gro_path):
    # returns res_order (list of (resid, resname) in file order, first atom
    # seen per residue) and idx_by_resid_atom mapping (resid, atomname) to
    # the 1-based atom index
    with open(gro_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        raise ValueError("GRO file looks truncated")

    natoms = int(lines[1].strip())
    atom_lines = lines[2:2 + natoms]

    res_order = []
    seen = set()
    idx_by_resid_atom = {}

    for i, line in enumerate(atom_lines, start=1):
        resid = int(line[0:5].strip())
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        if resid not in seen:
            res_order.append((resid, resname))
            seen.add(resid)
        idx_by_resid_atom[(resid, atomname)] = i

    return res_order, idx_by_resid_atom


def build_dna_lists(res_order):
    dna_resids, dna_bases = [], []
    for resid, rn in res_order:
        base = base_letter_from_resname(normalize_resname(rn))
        if base is not None:
            dna_resids.append(resid)
            dna_bases.append(base)
    if not dna_resids:
        raise ValueError("No DA/DT/DG/DC residues found in the gro")
    return dna_resids, dna_bases


def lookup_idx(idx_by_resid_atom, resid, aname):
    # try the exact name first, then case variants, then a slow scan
    # in case there's trailing whitespace weirdness in the gro
    if (resid, aname) in idx_by_resid_atom:
        return idx_by_resid_atom[(resid, aname)]
    if (resid, aname.upper()) in idx_by_resid_atom:
        return idx_by_resid_atom[(resid, aname.upper())]
    if (resid, aname.lower()) in idx_by_resid_atom:
        return idx_by_resid_atom[(resid, aname.lower())]
    for (r, a), idx in idx_by_resid_atom.items():
        if r == resid and a.strip().upper() == aname.upper():
            return idx
    raise KeyError(f"couldn't find atom '{aname}' on resid {resid}")


def read_pairs_dnaorder(pairs_path):
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))
    if not pairs:
        raise ValueError("no pairs found in pairs.tsv")
    return pairs


def build_triplet_and_pair(o1, o2, dna_resids, dna_bases, idx_by_resid_atom):
    n = len(dna_resids)
    if not (1 <= o1 <= n and 1 <= o2 <= n):
        raise ValueError(f"pair ({o1},{o2}) is outside DNA range 1..{n}")

    r1, b1 = dna_resids[o1 - 1], dna_bases[o1 - 1]
    r2, b2 = dna_resids[o2 - 1], dna_bases[o2 - 1]

    donors, acceptors = {"G", "T"}, {"A", "C"}
    if b1 in donors and b2 in acceptors:
        donor_res, donor_base, accept_res, accept_base = r1, b1, r2, b2
    elif b2 in donors and b1 in acceptors:
        donor_res, donor_base, accept_res, accept_base = r2, b2, r1, b1
    else:
        raise ValueError(f"pair {o1}({b1})-{o2}({b2}) isn't a valid donor/acceptor combo")

    if donor_base == "G":
        if accept_base != "C":
            raise ValueError(f"G donor needs a C acceptor, got {accept_base}")
        H = lookup_idx(idx_by_resid_atom, donor_res, "H1")
        D = lookup_idx(idx_by_resid_atom, donor_res, "N1")
        A = lookup_idx(idx_by_resid_atom, accept_res, "N3")
        return (H, D, A), (H, A)

    # donor_base == "T"
    if accept_base != "A":
        raise ValueError(f"T donor needs an A acceptor, got {accept_base}")
    H = lookup_idx(idx_by_resid_atom, donor_res, "H3")
    D = lookup_idx(idx_by_resid_atom, donor_res, "N3")
    A = lookup_idx(idx_by_resid_atom, accept_res, "N1")
    return (H, D, A), (H, A)


def write_ndx(out_path, triplets, pairs):
    with open(out_path, "w", encoding="utf-8") as f:
        for k, (h, d, a) in enumerate(triplets, start=1):
            f.write(f"[ BP{str(k).zfill(2)} ]\n")
            f.write(f"{h:6d} {d:6d} {a:6d}\n\n")

        f.write("[ BP_all ]\n")
        for h, d, a in triplets:
            f.write(f"{h:6d} {d:6d} {a:6d}\n")
        f.write("\n")

        for k, (h, a) in enumerate(pairs, start=1):
            f.write(f"[ HA{str(k).zfill(2)} ]\n")
            f.write(f"{h:6d} {a:6d}\n\n")

        f.write("[ HA_all ]\n")
        for h, a in pairs:
            f.write(f"{h:6d} {a:6d}\n")
        f.write("\n")


def main():
    if len(sys.argv) != 4:
        print("Usage: python make_wc_pairs_ndx.py structure.gro pairs.tsv output.ndx")
        sys.exit(1)

    gro_path = Path(sys.argv[1])
    pairs_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    res_order, idx_by_resid_atom = parse_gro_atoms(gro_path)
    dna_resids, dna_bases = build_dna_lists(res_order)
    pairs_dna = read_pairs_dnaorder(pairs_path)

    max_pair = max(max(a, b) for a, b in pairs_dna)
    if max_pair > len(dna_resids):
        raise ValueError(f"pairs.tsv references DNA index {max_pair} but only {len(dna_resids)} DNA residues found")

    triplets, pairs = [], []
    for o1, o2 in pairs_dna:
        t, p = build_triplet_and_pair(o1, o2, dna_resids, dna_bases, idx_by_resid_atom)
        triplets.append(t)
        pairs.append(p)

    write_ndx(out_path, triplets, pairs)
    print(f"wrote {len(triplets)} BP triplets and {len(pairs)} HA pairs to {out_path}")
    print("groups: BP01.., BP_all, HA01.., HA_all")


if __name__ == "__main__":
    main()
