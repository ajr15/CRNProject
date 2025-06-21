import json
from typing import List
import numpy as np
from utils import DummyReaction, DummySpecie, NasaPolynomial, build_rxn_tree, dummy_reactions_to_graph, add_atomization_energies

def read_species_from_list(species_list: List[str]):
    res = []
    for s in species_list:
        s = s.strip()
        if not "M" in s and len(s) > 0 and not s in ["AR", "HE"]:
            if s.endswith("("):
                s = s[:-1].strip()
            if s[:1].isdigit():
                n = int(s[0])
                for _ in range(n):
                    res.append(s[1:])
            else:
                res.append(s)
    return res

def parse_chemkin_thermo_block(lines):
    if len(lines) != 4:
        raise ValueError("Expected 4 lines for a NASA species block")
    line1 = lines[0]
    name = line1[:18].strip().split()[0]  # Get the first word as the name
    T_low = float(line1[45:55])
    T_high = float(line1[55:65])
    T_common = float(line1[65:75])
    coeffs = []
    for line in lines[1:]:
        for i in range(0, 75, 15):
            part = line[i:i + 15].strip().replace(" ", "").replace("D", "E")
            if part:
                coeffs.append(float(part))
    coeffs = coeffs[:14] # NASA format expects 14 coefficients
    if len(coeffs) != 14:
        raise ValueError(f"Expected 14 coefficients, got {len(coeffs)}")
    high_coeffs = coeffs[:7]
    low_coeffs = coeffs[7:]
    poly = NasaPolynomial(T_low, T_high, T_common, low_coeffs, high_coeffs)
    return DummySpecie(name, 0, {"thermo": poly.to_dict()})

def read_chemkin_thermo_file(path) -> List[DummySpecie]:
    species = []
    with open(path, 'r') as f:
        lines = f.readlines()
    # Strip whitespace
    lines = [line.rstrip('\n') for line in lines]
    # Skip optional global temperature line and top line
    i = 2
    # Process species blocks in chunks of 4 lines
    while True:
        block = lines[i:i+4]
        if any("END" in line for line in block):
            break
        try:
            sp = parse_chemkin_thermo_block(block)
            species.append(sp)
        except Exception as e:
            print(f"Warning: failed to parse block at lines {i+1}-{i+4}: {e}")
        i += 4
    return species

def read_all_reactions(path: str):
    """Read all reactions from USC Mech v2"""
    with open(path, "r") as f:
        reactions_block = False
        reactions = set()
        for line in f.readlines():
            # skip comment lines
            if line.startswith("!"):
                continue
            # start reaction block
            if "REACTIONS" in line:
                reactions_block = True
                continue
            # end reaction block
            if "END" in line and reactions_block:
                reactions_block = False
                continue
            # parse reaction line
            # make sure its a reaction line
            if "=" in line:
                if "!" in line:
                    line = line.split("!")[0]
                    if not "=" in line:
                        continue
                # make all reaction notations uniform
                line = line.replace(" + ", "+")
                line = line.replace("+ ", "+")
                line = line.replace(" +", "+")
                splited = line.split()
                activation = float(splited[-1])
                beta = float(splited[-2])
                pre_exp = float(splited[-3])
                forward_rxn_props = {"Ea": activation, "beta": beta, "A": pre_exp}
                reverse_rxn_props = {"rEa": activation, "rbeta": beta, "rA": pre_exp}
                if "=>" in line and not "<=>" in line: # non-reversible reaction
                    ajr = line.split("=>")
                    rs = read_species_from_list(ajr[0].split("+"))
                    ps = read_species_from_list(ajr[-1].split()[0].split("+"))
                    # append forward reaction - non reversible reaction
                    reactions.add(DummyReaction(rs, ps, properties=forward_rxn_props))
                else: # "=" or "<=>" in reaction - reversible
                    if "<=>" in line:
                        ajr = line.split("<=>")
                    elif " = " in line:
                        ajr = line.split(" = ")
                    else:
                        ajr = line.split("=")
                    rs = read_species_from_list(ajr[0].split("+"))
                    ps = read_species_from_list(ajr[-1].split()[0].split("+"))
                    # append forward reaction - with kinetic data
                    reactions.add(DummyReaction(rs, ps, properties=forward_rxn_props))
                    # append backward reaction - without kinetic data
                    reactions.add(DummyReaction(ps, rs, properties=reverse_rxn_props))
    return list(reactions)

def read_source_species(directory: str) -> List[str]:
    """Read source species from a file in the given directory"""
    with open("{}/source_species.txt".format(directory), "r") as f:
        source_species = f.readlines()
    source_species = [s.strip() for s in source_species if len(s.strip()) > 0]
    return source_species

def read_network(source: str):
    db_path = "../joined.db"
    print("reading mechanism from {}...".format(source))
    source_sps = read_source_species(source)
    species = read_chemkin_thermo_file("{}/thermo.chemkin".format(source))
    print("total number of species:", len(species))
    rxns = read_all_reactions("{}/kinetics.chemkin".format(source))
    print("total number of reactions:", len(rxns))
    print("building reaction graph with {} source species...".format(" + ".join(source_sps)))
    g = build_rxn_tree(rxns, source_sps)
    g = dummy_reactions_to_graph(g, species, source_sps)
    with open(source + "/smiles.json", "r") as f:
        symbols_to_smiles = json.load(f)
    g = add_atomization_energies(db_path, g, symbols_to_smiles)
    print("n reactions:", g.get_n_reactions())
    print("n species:", g.get_n_species())
    g.save("./{}.rxn".format(source))

if __name__ == "__main__":
    read_network("ammonia")
    exit()
    read_network("hydrogen")
    read_network("methane")

    print("Testing read")
    import torinanet as tn
    r = tn.core.RxnGraph.from_file("./ammonia.rxn")
    print("ammonia", "n reactions:", r.get_n_reactions(), "n species:", r.get_n_species())
    r = tn.core.RxnGraph.from_file("./hydrogen.rxn")
    print("hydrogen", "n reactions:", r.get_n_reactions(), "n species:", r.get_n_species())
    r = tn.core.RxnGraph.from_file("./methane.rxn")
    print("methane", "n reactions:", r.get_n_reactions(), "n species:", r.get_n_species())
