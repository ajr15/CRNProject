import pandas as pd
import json
from typing import List
import numpy as np
from utils import DummyReaction, DummySpecie, build_rxn_tree, dummy_reactions_to_graph, add_atomization_energies
import torinanet as tn
from src.NasaPolynomial import NasaPolynomial

R = 8.314462618 # universal gas constant
CAL_TO_J = 4.184 # conversion between calories to jouls

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

def specie_enthalpy(sp: tn.core.Specie, temperature: float):
    if not "thermo" in sp.properties:
        return None
    s = sp.properties["thermo"]
    poly = NasaPolynomial(**s)
    return poly.h(temperature)

def specie_entropy(sp: tn.core.Specie, temperature: float):
    if not "thermo" in sp.properties:
        return None
    s = sp.properties["thermo"]
    poly = NasaPolynomial(**s)
    return poly.s(temperature)

def calculate_reaction_energy(rxn: tn.core.Reaction, specie_energy_func):
    """
    Add reaction energy based on atomization energy, species enthalpy and gibbs free energy. 
    """
    products_e = [specie_energy_func(s) for s in rxn.products]
    reactants_e = [specie_energy_func(s) for s in rxn.reactants]
    if None in products_e or None in reactants_e:
        return None
    return sum(products_e) - sum(reactants_e)

def estimate_reaction_kinetics(rxn_graph: tn.core.RxnGraph, default_temperature: float=600):
    """
    Estimate reaction rate parameters
    """
    for rxn in rxn_graph.reactions:
        if not pd.isna(rxn.properties.get("rEa", pd.NA)) and \
            not pd.isna(rxn.properties.get("rA", pd.NA)) and \
            not pd.isna(rxn.properties.get("rbeta", pd.NA)):
            h = calculate_reaction_energy(rxn, lambda s: specie_enthalpy(s, default_temperature))
            s = calculate_reaction_energy(rxn, lambda s: specie_entropy(s, default_temperature))
            dn = calculate_reaction_energy(rxn, lambda s: 1)
            if h is None or s is None:
                continue
            # note that we calculate using the reverse reaction h and s, so we need to negate them
            rxn.properties["Ea"] = h + rxn.properties["rEa"] * CAL_TO_J
            rxn.properties["A"] = rxn.properties["rA"] * np.exp(s / R) * 1 / R ** dn
            rxn.properties["beta"] = rxn.properties["rbeta"] + dn
        elif not pd.isna(rxn.properties.get("Ea", pd.NA)):
            # fix Ea to be in J/mol
            rxn.properties["Ea"] = rxn.properties["Ea"] * CAL_TO_J
        # calculate the logk via arrhenius relation
        rxn.properties["logk"] = np.log(rxn.properties["A"]) + rxn.properties["beta"] * np.log(default_temperature) - rxn.properties["Ea"] / (R * default_temperature)
    return rxn_graph

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
    g = estimate_reaction_kinetics(rxn_graph=g, default_temperature=600)
    print("n reactions:", g.get_n_reactions())
    print("n species:", g.get_n_species())
    g.save("./{}.rxn".format(source))

if __name__ == "__main__":
    read_network("ammonia")
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
