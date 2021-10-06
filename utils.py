import logging
import os
import re
import time
import glob
import pandas as pd

from graphstructure import GraphStructure
from kidney_digraph import KidneyReadException, Digraph
from kidney_ndds import NddEdge, Ndd

LOG_FORMAT = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"


def get_logger(logfile=None):
    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logger


def add_uniform_probabilities(graph, p_reject, p_success_accept, p_success_noquery):
    for e in graph.all_edge_list:
        e.p_reject = p_reject
        e.p_success_accept = p_success_accept
        e.p_success_noquery = p_success_noquery


def read_new_format_unos_graph(edge_filename, cycle_cap, chain_cap, logger=None):
    df_edges = pd.read_csv(edge_filename)

    expected_columns = [
        "KPD_Match_Run_ID",
        "Candidate_pt_code",
        "Candidates_pair_pt_code",
        "Donor_pt_code",
        "Donors_pair_pt_code",
        "Total_Weight",
        "non_directed",
    ]

    if not len(expected_columns) == len(df_edges.columns):
        raise KidneyReadException(
            f"Edgeweights file {edge_filename} has {len(df_edges.columns)} columns. "
            f"Expected {len(expected_columns)}."
        )

    for i_col, expected in enumerate(expected_columns):
        if not simple_string(expected) == simple_string(df_edges.columns[i_col]):
            raise KidneyReadException(
                f"Column {(i_col + 1)} in *edgeweights.csv should be {simple_string(expected)}."
                f"Instead we found column {simple_string(df_edges.columns[i_col])}."
            )

    col_names = [
        "match_run",
        "patient_id",
        "patient_pair_id",
        "donor_id",
        "donor_paired_patient_id",
        "weight",
        "non_directed",
    ]

    df_edges.columns = col_names

    # nonzero_edges = df_edges.loc[df_edges["weight"] > 0]
    kpd_edges = df_edges.loc[(df_edges["weight"] > 0) & (df_edges["non_directed"] == 0)]

    vtx_id = set(
        list(kpd_edges["patient_id"].unique())
        + list(kpd_edges["donor_paired_patient_id"].unique())
    )
    vtx_count = len(vtx_id)
    digraph = Digraph(vtx_count)

    # vtx_index[id] gives the index in the digraph
    vtx_index = dict(zip(vtx_id, range(len(vtx_id))))

    warned = False
    for index, row in kpd_edges.iterrows():
        src_id = vtx_index[row["donor_paired_patient_id"]]
        tgt_id = vtx_index[row["patient_id"]]
        weight = row["weight"]
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException(f"Vertex index {src_id} out of range.")
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException(f"Vertex index {tgt_id} out of range.")
        if src_id == tgt_id:
            raise KidneyReadException(
                f"Self-loop from {src_id} to {src_id} not permitted"
            )
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]) & ~warned:
            print(f"# WARNING: Duplicate edge in file: {edge_filename}")
            warned = True
        if weight == 0:
            raise KidneyReadException(f"Zero-weight edge from {src_id} to {tgt_id}")

        digraph.add_edge(
            weight,
            digraph.vs[src_id],
            digraph.vs[tgt_id],
            edge_data={"donor_id": row["donor_id"], "patient_id": row["patient_id"]},
        )

    ndd_edges = df_edges.loc[(df_edges["weight"] > 0) & (df_edges["non_directed"] == 1)]
    ndd_id = set(list(ndd_edges["donor_id"].unique()))

    ndd_count = len(ndd_id)

    if ndd_count > 0:
        ndd_list = [Ndd(id=i) for i in range(ndd_count)]
        ndd_index = dict(
            zip(ndd_id, range(len(ndd_id)))
        )  # ndd_index[id] gives the index in the digraph

        # Keep track of which edges have been created already, to detect duplicates
        edge_exists = [[False for v in digraph.vs] for ndd in ndd_list]

        for index, row in ndd_edges.iterrows():
            src_id = ndd_index[row["donor_id"]]
            tgt_id = vtx_index[row["patient_pair_id"]]
            weight = row["weight"]
            if src_id < 0 or src_id >= ndd_count:
                raise KidneyReadException(f"NDD index {src_id} out of range.")
            if tgt_id < 0 or tgt_id >= digraph.n:
                raise KidneyReadException(f"Vertex index {tgt_id} out of range.")

            ndd_list[src_id].add_edge(
                NddEdge(
                    digraph.vs[tgt_id],
                    weight,
                    src_id=ndd_list[src_id].id,
                    src=ndd_list[src_id],
                    data={"donor_id": row["donor_id"], "patient_id": row["patient_id"]},
                )
            )
            edge_exists[src_id][tgt_id] = True
    else:
        ndd_list = []

    print("ndd count", len(ndd_list))
    graph = GraphStructure(
        digraph, ndd_list, cycle_cap, chain_cap, name=edge_filename, logger=logger
    )
    for e in graph.all_edge_list:

        e.data["patient_ctr"] = 0
        e.data["donor_ctr"] = 0

    return graph

# ----------------------------------------------------------------------------------------------------------------------
# non-kidney-exchange helper functions
# ----------------------------------------------------------------------------------------------------------------------


def simple_string(complex_string, non_numeric=False):
    if non_numeric:
        return re.sub(r"[^A-Za-z]+", "", complex_string).lower()
    else:
        return re.sub(r"[^A-Za-z0-9]+", "", complex_string).lower()


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + "_%s." + extension) % timestr
    return os.path.join(output_dir, output_string)
