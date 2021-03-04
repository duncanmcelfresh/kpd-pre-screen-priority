import argparse
import os

from greedy_edge_choice import greedily_choose_edge_list_timelimit
from kidney_graph_io import read_unos_graph_with_data, read_unos_graph
from utils import get_logger, generate_filepath
import numpy as np


def calculate_priority(args):
    """
    read files from a UNOS KPD match run, and write a file with prioritization scores for edge pre-screening

    edge prioritization scores are calculated as follows.
    1) set the score of all unmatchable edges to -1
    2) assume N edges will be pre-screened in total
    3) run the Greedy algorithm to select N edges
    4) set the score of the i^th edge selected by greedy to be N - i + i.
        the first-selected edge has score N, the second has score N - 1, and so on.
    5) all other matchable edges have score 0

    In other words, N is the number of edges that will have a non-zero score, so N should be large.

    NOTE:
        - the output file only includes edges with weight > 0. all zero-weight edges are ignored.
        - the output file only contains one row for each edge, even if the input *edgeweights.csv file
            has duplicate edges.

    INPUT: args (argparse). required arg fields:
    - num_prescreen_edges (int): number of edges we assume will be pre-screened. This should be large (default = 200).
    - out_dir (str): output directory for writing pre-screen prioritization file
    - kpd_dir (str): directory containing UNOS KPD match run files. must contain one file with name suffix *edgeweights.csv
    - cycle_cap (int): cycle cap for KPD match run
    - chain_cap (int): chain cap for KPD match run
    - seed (ind): random seed for the greedy algorithm
    - num_leaf_samples (int): used by edge_selection_tree.evaluate_edge_list. should be fairly large (>200),
        if num_prescreen_edges is large.
    - max_level _for_pruning (int): used by edge_selection_tree.get_candidate_edges. if this is >= num_prescreen_edges,
        roll out the entire search tree. larger values require exponentially more memory. recommend <10.
    - time_limit (int): time limit for the greedy alg in seconds
    - edge_success_prob (float): edge success probability used in simulations
    """

    logger = get_logger(logfile=generate_filepath(args.out_dir, "LOGS", "txt"))
    rs = np.random.RandomState(args.seed)

    graph = read_unos_graph(args.kpd_dir, args.cycle_cap, args.chain_cap)

    # calculate edge importance using the greedy heuristic
    logger.info("running greedy heuristic")
    graph.init_optconfig(edge_success_prob=args.edge_success_prob)
    greedy_edges = greedily_choose_edge_list_timelimit(
        graph,
        args.num_prescreen_edges,
        rs,
        args.time_limit,
        args.num_leaf_samples,
        args.max_level_for_pruning,
        logger,
    )

    # initialize edge scores to -1 (unmatchable)
    edge_score_dict = {edge: -1 for edge in graph.all_edge_list}

    # initialize matchable edges to 0.0 (low priority, but matchable)
    for e in graph.matchable_edge_list:
        edge_score_dict[e] = 0

    # assign positive scores to all edges selected by the greedy algorithm
    for i, e in enumerate(greedy_edges):
        assert e.matchable
        edge_score_dict[e] = len(greedy_edges) - i

    # write csv
    out_file = generate_filepath(args.out_dir, "prescreen_priority", "csv")

    with open(out_file, "w") as f:
        f.write("KPD_candidate_id,KPD_donor_id,prescreen_score\n")
        for e, score in edge_score_dict.items():
            f.write(f"{e.data['donor_id']},{e.data['patient_id']},{int(score)}\n")

    logger.info("done.")


def parse_args():
    """
    - num_prescreen_edges (int): number of edges we assume will be pre-screened. This should be large (default = 200).
    - out_dir (str): output directory for writing pre-screen prioritization file
    - kpd_dir (str): directory containing UNOS KPD match run files. must contain one file with name suffix *edgeweights.csv
    - cycle_cap (int): cycle cap for KPD match run
    - chain_cap (int): chain cap for KPD match run
    - seed (ind): random seed for the greedy algorithm
    - num_leaf_samples (int): used by edge_selection_tree.evaluate_edge_list. should be fairly large (>200),
        if num_prescreen_edges is large.
    - max_level_for_pruning (int): used by edge_selection_tree.get_candidate_edges. if this is >= num_prescreen_edges,
        roll out the entire search tree. larger values require exponentially more memory. recommend <10.
    - time_limit (int): time limit for the greedy alg in seconds
    - edge_success_prob (float): edge success probability used in simulations
    """
    parser = argparse.ArgumentParser()

    # example: --num-prescreen-edges 200 --out-dir ./output --kpd-dir /Users/duncan/research/graphs/kpd_zips/zips/KPD_CSV_IO_20160602

    parser.add_argument(
        "--num-prescreen-edges",
        type=int,
        default=200,
        help="number of edges we assume will be pre-screened. This should be large.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, help="directory for output",
    )
    parser.add_argument(
        "--kpd-dir",
        type=str,
        default=None,
        help="directory containing UNOS KPD match run files.  must contain one file with name suffix *edgeweights.csv",
    )
    parser.add_argument("--chain-cap", type=int, default=4, help="chain cap")
    parser.add_argument("--cycle-cap", type=int, default=3, help="cycle cap")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--num-leaf-samples",
        type=int,
        default=200,
        help="used by edge_selection_tree.evaluate_edge_list. should be fairly large (>200), "
        "if num_prescreen_edges is large",
    )
    parser.add_argument(
        "--max-level-for-pruning",
        type=int,
        default=4,
        help="used by edge_selection_tree.get_candidate_edges. if this is >= num_prescreen_edges, "
        "roll out the entire search tree. larger values require exponentially more memory. recommend <10.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=3600,
        help="time limit for the greedy alg in seconds",
    )
    parser.add_argument(
        "--edge-success-prob",
        type=float,
        default=0.5,
        help="edge success probability used in simulations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calculate_priority(args)
