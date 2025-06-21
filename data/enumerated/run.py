import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
import argparse
import enumerator
import torinanet as tn
import torinax as tx

DEFAULT_KWDS = {
    "ac_filters": [
        tn.iterate.ac_matrix_filters.MaxAtomsOfElement({7: 2, 8: 2}), 
        tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
        tn.iterate.ac_matrix_filters.MaxComponents(2),
        tn.iterate.ac_matrix_filters.MaxRingNumber(0),
    ],
    "reaction_energy_th": 0.15,
    "sp_energy_th": 0.25,
    "molrank_temperature": 600,
    "use_molrank": True,
    "use_leaf": True
}

COMP_ARGS = {
    "nh3+o2": {
        "reactants": [tn.core.Specie("O=O"), tn.core.Specie("N")],
        "kwds": {}
    },
    "nh3+o2+h2": {
        "reactants": [tn.core.Specie("O=O"), tn.core.Specie("N"), tn.core.Specie("[H][H]")],
        "kwds": {}
    }
}


if __name__ == "__main__":
    # argument parser for different netnworks
    parser = argparse.ArgumentParser()
    parser.add_argument("reaction", type=str, help="reaction to enumerate. options: {}".format(", ".join(COMP_ARGS.keys())))
    parser.add_argument("--rundir", type=str, default=None, help="directory to make the run in")
    parser.add_argument("--n_iterations", type=int, default=8, help="directory to make the run in")
    # add all controllable parameters from default kwds
    for kw, val in DEFAULT_KWDS.items():
        if not kw == "ac_filters":
            parser.add_argument(kw, type=type(val), default=val)
    args = parser.parse_args()
    # init
    if args.rundir is None:
        args.rundir = "./{}".format(args.reaction)
    if not os.path.isdir(args.rundir):
        os.mkdir()
    os.chdir(args.rundir)
    slurm_client = tx.clients.SlurmClient(4, "8GB", args.reaction)
    dask_client = None
    rxn_graph = tn.core.RxnGraph()
    # setting source species in graph
    rxn_graph.set_source_species(COMP_ARGS[args.reaction], force=True)
    # get comp kwds
    kwds = DEFAULT_KWDS.copy()
    kwds.update(COMP_ARGS[args.reaction])
    # update kwds from command line arguments
    for arg, val in kwds.items():
        argval = getattr(args, arg)
        if argval != val:
            kwds[arg] = argval
    # getting threshold
    print("RUNNING FOR {}".format(args.reaction))
    enumer = enumerator.SimpleEnumerator(rxn_graph, args.n_iterations, ".", slurm_client,
                                         reflect=True,
                                         **kwds[args.reaction])
    enumer.enumerate()
