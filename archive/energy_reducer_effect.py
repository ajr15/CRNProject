import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous, norm
from model import PreferentialAttachmentModel, SimulatedReactionGraph

def apply_energy_th(g: SimulatedReactionGraph, th: float, verbose: int=1):
    for rxn in g.reactions():
        if rxn.oid in g.g and rxn.properties["energy"] > th:
            g.remove_reaction(rxn.oid)
            if verbose > 0:
                print("species={} reactions={}".format(len(g.get_species()), len(g.get_reactions())))
    return g

def single_run(n_reactions: int, n_species: int, energy_th: float, sp_energy_dist: rv_continuous, verbose: int=1):
    model = PreferentialAttachmentModel(n_species=n_species, 
                                        n_reactions=n_reactions, 
                                        specie_energy_dist=sp_energy_dist)
    original = model.generate_graph()
    reduced = apply_energy_th(original.copy(), energy_th, verbose=verbose)
    return original, reduced

def energy_distribution(g: SimulatedReactionGraph, ax=None):
    ajr = []
    for r in g.reactions():
        if "energy" in r.properties:
            ajr.append(r.properties["energy"])
        else:
            print("I DONT HAVE ENERGY !!!!")
    if ax is None:
        ax = plt.gca()
    ax.hist(ajr)

def get_energies(g: SimulatedReactionGraph):
    sps = []
    for s in g.species():
        sps.append(s.properties["energy"])
    rxns = []
    for rxn in g.reactions():
        rxns.append(rxn.properties["energy"])
    return sps, rxns

def analyze_network(g: SimulatedReactionGraph, prefix: str="") -> dict:
    ajr = {}
    ajr[prefix + "species"] = len(g.get_species())
    ajr[prefix + "reactions"] = len(g.get_reactions())
    sps_e, rxns_e = get_energies(g)
    ajr[prefix + "avg_specie_e"] = np.mean(sps_e)
    ajr[prefix + "std_specie_e"] = np.std(sps_e)
    ajr[prefix + "avg_reaction_e"] = np.mean(rxns_e)
    ajr[prefix + "std_reaction_e"] = np.std(rxns_e)
    return ajr

def reduction_effect(n_reactions: list, specie_ratios: list):
    data = []
    for n_rxns in n_reactions:
        for ratio in specie_ratios:
            nspecies = int(round(n_rxns * ratio))
            print("Running for {} reactions and {} speices".format(n_rxns, nspecies))
            og, rg = single_run(n_rxns, nspecies, 0.0, norm(0.4, 0.3))
            d = analyze_network(og, "original_")
            d.update(analyze_network(rg, "reduced_"))
            data.append(d)
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = reduction_effect([1000, 3000, 5000], [0.1, 0.2, 0.3])
    df.to_csv("energy_reduction_effect.csv")
    exit()
    # model the reduction effect of different thresholds
    og, rg = single_run(1000, 150, 0.0, norm(0.4, 0.3))
    print("OG species={} reactions={}".format(len(og.get_species()), len(og.get_reactions())))
    print("RG species={} reactions={}".format(len(rg.get_species()), len(rg.get_reactions())))
    fig, axs = plt.subplots(ncols=2)
    plt.tight_layout()
    og_sps_e, og_rxns_e = get_energies(og)
    axs[0].hist(og_sps_e)
    axs[0].set_title("original")
    rg_sps_e, rg_rxns_e = get_energies(rg)
    axs[1].hist(rg_sps_e)
    axs[1].set_title("reduced")
    plt.show()