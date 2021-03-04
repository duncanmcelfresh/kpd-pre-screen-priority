import glob
import logging
import os
import pandas as pd

from graphstructure import GraphStructure
from kidney_digraph import Digraph, KidneyReadException
from kidney_ndds import Ndd, NddEdge
from utils import simple_string

FORMAT = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger()

NULL_KPD_DATA = None


def read_unos_graph(directory, cycle_cap, chain_cap):
    """read a unos-format exchange, and return a list of kidney_ndd.Ndd objects and a kidney_digraph.Digraph object.

    each unos-format exchange is contained in a subdirectory with the naming format 'KPD_CSV_IO_######'. Each exchange
     subdirectory must contain a file with name ########_edgeweights.csv
    """
    # look for edge files
    edge_files = glob.glob(os.path.join(directory, "*edgeweights.csv"))

    name = os.path.basename(directory)

    # there should only be one edgeweights file
    if not len(edge_files) == 1:
        raise KidneyReadException(
            f"Directory {directory} contains {len(edge_files)} edgeweights files. "
            f"Only one expected."
        )

    edge_filename = edge_files[0]

    df = pd.read_csv(edge_filename)

    expected_columns = [
        "KPD Match Run ID",
        "KPD Candidate ID",
        "Candidate's KPD Pair ID",
        "KPD Donor ID",
        "Donor's KPD Pair ID",
        "Total Weight",
    ]

    if not len(expected_columns) == len(df.columns):
        raise KidneyReadException(
            f"Edgeweights file {edge_filename} has {len(df.columns)} columns. "
            f"Expected {len(expected_columns)}."
        )

    for i_col, expected in enumerate(expected_columns):
        if not simple_string(expected) == simple_string(df.columns[i_col]):
            raise KidneyReadException(
                f"Column {(i_col + 1)} in *edgeweights.csv should be {simple_string(expected)}."
                f"Instead we found column {simple_string(df.columns[i_col])}."
            )

    col_names = [
        "match_run",
        "patient_id",
        "patient_pair_id",
        "donor_id",
        "donor_paired_patient_id",
        "weight",
    ]

    df.columns = col_names

    # last column is edge weights -- only take nonzero edges
    nonzero_edges = df.loc[df["weight"] > 0]

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

    graph = GraphStructure(digraph, ndd_list, cycle_cap, chain_cap, name=name)

    return graph


def read_unos_graph_with_data(directory, cycle_cap, chain_cap):
    """
    read a unos-format exchange, and return a list of kidney_ndd.Ndd objects and a kidney_digraph.Digraph object, and
    save donor/recipient data

    each unos-format exchange is contained in a subdirectory with the naming format 'KPD_CSV_IO_######'. Each exchange
     subdirectory must contain files of the format:
      - *edgeweights.csv
      - *donor.csv
      - *recipient.csv
    """

    if directory.endswith(os.sep):
        name = os.path.basename(directory[:-1])
    else:
        name = os.path.basename(directory)

    # look for  files
    edge_files = glob.glob(os.path.join(directory, "*edgeweights.csv"))
    donor_files = glob.glob(os.path.join(directory, "*donor.csv"))
    recip_files = glob.glob(os.path.join(directory, "*recipient.csv"))

    # there should only be one of each file
    if not len(donor_files) == 1:
        raise KidneyReadException(
            "There should be one *donor.csv file in the directory. "
            f"There are {len(donor_files)} in {directory}."
        )
    if not len(recip_files) == 1:
        raise KidneyReadException(
            "There should be one *recipient.csv file in the directory. "
            f"There are {len(recip_files)} in {directory}."
        )
    if not len(edge_files) == 1:
        raise KidneyReadException(
            "There should be one *edgeweights.csv file in the directory. "
            f"There are {len(edge_files)} in {directory}."
        )

    donor_file = donor_files[0]
    recip_file = recip_files[0]
    edge_filename = edge_files[0]

    df_donor = pd.read_csv(donor_file)
    df_recip = pd.read_csv(recip_file)

    # make all cols lowercase
    df_donor.columns = [c.lower() for c in df_donor.columns]
    df_recip.columns = [c.lower() for c in df_recip.columns]

    # if no cpra col, then add null values
    if "cpra" not in df_recip.columns:
        logger.info("CPRA column not found")
        df_recip["cpra"] = NULL_KPD_DATA

    # add columns for missing data if they don't exist
    # if no cpra col, then add null values
    if "cpra" not in df_recip.columns:
        logger.info("COL NOT FOUND: cpra")
        df_recip["cpra"] = NULL_KPD_DATA

    if "highly_sensitized" not in df_recip.columns:
        logger.info("COL NOT FOUND: highly_sensitized")
        df_recip["highly_sensitized"] = NULL_KPD_DATA

    if "abo" not in df_recip.columns:
        if "abo blood group" in df_recip.columns:
            df_recip["abo"] = df_recip["abo blood group"]
        else:
            raise Exception("no abo column found")

    # validate donor data
    if not "abo" in df_donor.columns:
        raise KidneyReadException(f"Donor file {donor_file} does not have ABO column.")

    # validate recip data
    if not "abo" in df_recip.columns:
        raise KidneyReadException(
            f"Recipient file {recip_file} does not have ABO column."
        )

    if not "cpra" in df_recip.columns:
        raise KidneyReadException(
            f"Recipient file {recip_file} does not have CPRA column."
        )

    if not "highly_sensitized" in df_recip.columns:
        raise KidneyReadException(
            f"Recipient file {recip_file} does not have highly-sensitized."
        )

    # remove abo subtypes and make lowercase
    df_donor["abo"] = df_donor["abo"].apply(
        lambda x: simple_string(x, non_numeric=True)
    )
    df_recip["abo"] = df_recip["abo"].apply(
        lambda x: simple_string(x, non_numeric=True)
    )

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
                f"Column {(i_col+1)} in *edgeweights.csv should be {simple_string(expected)}."
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

    # vtx_index[id] gives the index in the digraph
    vtx_count = len(vtx_id)
    vtx_index = dict(zip(vtx_id, range(len(vtx_id))))
    vtx_index_to_id = {v: k for k, v in vtx_index.items()}

    digraph = Digraph(vtx_count, aux_vertex_id=vtx_index_to_id)

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

        # for the donor pair, add the the donor ID to the vertex's list of donor IDs unless it's already there
        digraph.vs[src_id].donor_set.add(row["donor_id"])

        digraph.add_edge(
            weight, digraph.vs[src_id], digraph.vs[tgt_id], edge_data=row.to_dict()
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

        for k, v in ndd_index.items():
            ndd_list[v].aux_id = k

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

    graph = GraphStructure(
        digraph,
        ndd_list,
        cycle_cap,
        chain_cap,
        name=name,
        df_donor=df_donor,
        df_recip=df_recip,
    )

    return graph
