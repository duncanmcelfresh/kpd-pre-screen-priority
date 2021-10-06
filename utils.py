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


def read_unos_graph(directory, cycle_cap, chain_cap, logger=None):
    """read a unos-format exchange, and return a list of kidney_ndd.Ndd objects and a kidney_digraph.Digraph object.

    each unos-format exchange is contained in a subdirectory with the naming format 'KPD_CSV_IO_######'. Each exchange
     subdirectory must contain a file with name ########_edgeweights.csv
    """
    name = os.path.basename(directory)

    # --- read edge data ---

    # look for edge files - there should be only one
    edge_files = glob.glob(os.path.join(directory, "*edgeweights.csv"))

    # there should only be one edgeweights file
    if not len(edge_files) == 1:
        raise KidneyReadException(
            f"Directory {directory} contains {len(edge_files)} edgeweights files. "
            f"Only one expected."
        )
    edge_filename = edge_files[0]

    df_edges = pd.read_csv(edge_filename)

    expected_columns = [
        "KPD Match Run ID",
        "KPD Candidate ID",
        "Candidate's KPD Pair ID",
        "KPD Donor ID",
        "Donor's KPD Pair ID",
        "Total Weight",
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
    ]

    df_edges.columns = col_names

    # --- read donor data ---

    # look for donor files - there should be one
    donor_files = glob.glob(os.path.join(directory, "*donor.csv"))

    # there should only be one donor file
    if not len(donor_files) == 1:
        raise KidneyReadException(
            f"Directory {directory} contains {len(donor_files)} donor files. "
            f"Only one expected."
        )

    donor_filename = donor_files[0]

    df_donor = pd.read_csv(donor_filename)
    df_donor.columns = [simple_string(c) for c in df_donor.columns]

    # only two fields are needed from donor data
    required_donor_fields = [
        "kpddonorid",
        "homectr",
    ]

    for expected_col in required_donor_fields:
        if expected_col not in df_donor.columns:
            raise KidneyReadException(f"Column {expected_col} not found in *donor.csv")

    # read recip id and transplant center into a dict
    donor_ctr = {}
    for index, row in df_donor.iterrows():
        if row["kpddonorid"] in donor_ctr:
            logger.info(
                f"duplicate donors with id {row['kpddonorid']}, skipping the duplicates"
            )
        else:
            donor_ctr[row["kpddonorid"]] = row["homectr"]

    # --- read recipient data ---

    # look for recipient files - there should be one
    recip_files = glob.glob(os.path.join(directory, "*recipient.csv"))

    # there should only be one recip file
    if not len(recip_files) == 1:
        raise KidneyReadException(
            f"Directory {directory} contains {len(recip_files)} recipient files. "
            f"Only one expected."
        )

    recip_filename = recip_files[0]

    df_recip = pd.read_csv(recip_filename)
    df_recip.columns = [simple_string(c) for c in df_recip.columns]

    # only two fields are needed from recip data
    required_recip_fields = [
        "kpdcandidateid",
        "transplantctr",
    ]

    for expected_col in required_recip_fields:
        if not expected_col in df_recip.columns:
            raise KidneyReadException(
                f"Column {expected_col} not found in *recipient.csv"
            )

    # read recip id and transplant center into a dict
    recip_ctr = {}
    for index, row in df_recip.iterrows():
        if row["kpdcandidateid"] in recip_ctr:
            logger.info(
                f"duplicate recipients with id {row['kpdcandidateid']}, skipping the duplicates"
            )
        else:
            recip_ctr[row["kpdcandidateid"]] = row["transplantctr"]

    # --- data preparation ---

    # last column is edge weights -- only take nonzero edges
    nonzero_edges = df_edges.loc[df_edges["weight"] > 0]

    # remove NDD edges
    kpd_edges = nonzero_edges.loc[~nonzero_edges["donor_paired_patient_id"].isnull()]

    # get unique vertex ids
    # Note, in the *edgeweights.csv files:
    # - "KPD Candidate ID" (or "patient_id" here) is the patient/recipient's UNOS ID
    # - "Donor's KPD Pair ID" is the UNOS ID of the donor's associated patient (or None if the donor is an NDD)
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

    # now read NDDs - take only NDD edges
    ndd_edges = nonzero_edges.loc[nonzero_edges["donor_paired_patient_id"].isnull()]
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

    vtd_id_from_index = {val: key for key, val in vtx_index.items()}
    for i, v in enumerate(digraph.vs):
        # add vertex center to v.data

        # vtx_index[id] gives the index in the digraph
        vtx_index = dict(zip(vtx_id, range(len(vtx_id))))

    graph = GraphStructure(
        digraph, ndd_list, cycle_cap, chain_cap, name=name, logger=logger
    )

    # add donor and recip center data to each edge
    for e in graph.all_edge_list:
        if not e.data["donor_id"] in donor_ctr:
            raise KidneyReadException(f"no center for donor id {e.data['donor_id']}")
        if not e.data["patient_id"] in recip_ctr:
            raise KidneyReadException(
                f"no center for patient id {e.data['patient_id']}"
            )

        e.data["patient_ctr"] = recip_ctr[e.data["patient_id"]]
        e.data["donor_ctr"] = donor_ctr[e.data["donor_id"]]

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
