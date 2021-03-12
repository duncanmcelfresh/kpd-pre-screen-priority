# functions for selecting edges
import time
import numpy as np

from gurobipy import *
from kidney_ip import solve_picef_model


def evaluate_edge_list(edge_list, graph, num_leaf_samples, rs):
    """
    evaluate the expected outcome of this edge query list. if num_leaf_samples is large enough to completely estimate
    the mean, then call evaluate_edge_list_exact (i.e., without sampling). otherwise, evaluate by sampling.

    inputs:
    - edges (iterable(edge objects)) : list of edges to simulate realizations after querying
    - graph: GraphStructure object
    - num_leaf_samples: number of outcomes to sample
    """

    if num_leaf_samples >= 2 ** len(edge_list):
        return evaluate_edge_list_exact(edge_list, graph)

    weight_list = np.zeros(num_leaf_samples)
    prob_list = np.zeros(num_leaf_samples)
    edge_acceptance_list = []
    for i in range(num_leaf_samples):
        u_rvs = rs.uniform(0, 1, len(edge_list))
        p_reject_list = np.array([e.p_reject for e in edge_list])
        edge_acceptance = (
            u_rvs >= p_reject_list
        )  # if the uniform RV is greater than p_reject, then accept

        prob_list[i], weight_list[i] = eval_graph_on_outcome(
            edge_list, edge_acceptance, graph
        )
        edge_acceptance_list.append(edge_acceptance)

    return weight_list, prob_list, edge_acceptance_list


def evaluate_edge_list_exact(edge_list, graph):
    """
    exactly evaluate the mean and stddev of the outcome of this edge query list

    inputs:
    - edge_list (list(edge objects)) : list of edges to simulate realizations after querying
    - graph: GraphStructure object

    find the expected quality of this edge list, by enumerating all realizations of the queried edges (edge_list). for
    each realization:
    - calculate the probability of this realization
    - find the max-weight deterministic matching in this outcome (not using rejected edges)
    - calculate the expected weight in this outcome
    """

    # iterate through all reject/accpet outcomes for the selected edges
    # outcome[i] = True means that edge i is accepted, rejected if false
    num_outcomes = 2 ** len(edge_list)
    weight_list = np.zeros(num_outcomes)
    prob_list = np.zeros(num_outcomes)
    edge_acceptance_list = []
    for i, edge_acceptance in enumerate(
        itertools.product([True, False], repeat=len(edge_list))
    ):  # for every possible outcome

        prob_list[i], weight_list[i] = eval_graph_on_outcome(
            edge_list, edge_acceptance, graph
        )
        edge_acceptance_list.append(edge_acceptance)

    return weight_list, prob_list, edge_acceptance_list


def eval_graph_on_outcome(edge_list, edge_acceptance, graph):
    """
    evaluate the expected weight of a max-weight matching on a graph, given the outcome for a set of edges

    args:
    - edges (list(edge)): set of edges which are queried
    - edge_acceptance (list(bool)): if edge_acceptance[i] = True, then edges[i] is accepted, otherwise it is rejected.
    - graph (GrpahStructure): kidney exchange graph - must have an OptConfig object as a property
    """

    opt_solution = solve_picef_model(
        graph.optconfig,
        remove_edges=[e for e, accept in zip(edge_list, edge_acceptance) if not accept],
    )

    ## printing for debugging
    # print(f"matching edges: {[str(e) for e in opt_solution.matching_edges]}")
    # print(f"matching score: {sum(e.weight for e in opt_solution.matching_edges)}")

    # calculate the probability of this outcome occurring
    outcome_prob = np.prod(
        [
            (1.0 - e.p_reject) if accept else e.p_reject
            for e, accept in zip(edge_list, edge_acceptance)
        ]
    )

    # calculate total expected weight
    total_weight = 0.0

    # expected cycle weight
    for cycle in opt_solution.cycle_obj:
        cycle_weight = sum(e.weight for e in cycle.edges)
        cycle_success_prob = np.prod(
            [
                e.p_success_accept
                if e in edge_list
                else e.p_success_noquery  # if edges are matched and in edge_list, then they succeeded
                for e in cycle.edges
            ]
        )

        total_weight += cycle_weight * cycle_success_prob

    # expected chain weight
    for chain in opt_solution.chains:
        chain_success_prob = 1.0
        chain_edges = chain.get_edge_objs(graph.graph, graph.altruists)
        chain_weight = 0.0
        for e in chain_edges:
            chain_success_prob *= (
                e.p_success_accept if e in edge_list else e.p_success_noquery
            )
            chain_weight += e.weight * chain_success_prob

        total_weight += chain_weight

    return outcome_prob, total_weight


def get_candidate_edges(edge_list, graph, max_level_for_pruning):
    """
    get the set of potential edges to query, given that edge_list is already in the query set.

    if len(edge_list) <= max_level_for_pruning, then enumerate all fail/succeed outcomes for all edges in edge_list, and
    find the matched edges in each outcome - then return the union of these matched edges.

    otherwise, just return graph.matchable_edges
    """
    if len(edge_list) <= max_level_for_pruning:
        candidate_edges = set()
        for edge_fail_list in itertools.product([True, False], repeat=len(edge_list)):
            opt_solution = solve_picef_model(
                graph.optconfig,
                remove_edges=[
                    e for e, fail in zip(edge_list, edge_fail_list) if not fail
                ],
            )

            # update the set of potential child edges
            candidate_edges.update(opt_solution.matching_edges)
    else:
        # otherwise create a child for each matchable edge
        candidate_edges = set(graph.matchable_edge_list)

    return candidate_edges.difference(edge_list)


def greedily_choose_edge_list_timelimit(
    graph,
    num_edges,
    edge_selection_rs,
    time_limit,
    num_leaf_samples,
    max_level_for_pruning,
    logger,
):
    edge_list = []

    start_time = time.time()

    for i in range(num_edges):
        logger.info(f"running greedy step {(i + 1)} of {num_edges}")
        max_val = 0.0
        best_edge = None

        # get the set of candidate edges to add to edge_list (the first edge to add will be in this list)
        candidate_edges = get_candidate_edges(edge_list, graph, max_level_for_pruning)

        if len(candidate_edges) == 0:
            logger.info(f"greedy has no candidate edges at step {i + 1}")
            break

        for e in candidate_edges:

            # if we're out of time
            if (time.time() - start_time) > time_limit:

                # randomly add all remaining edges to sample set
                logger.info(
                    f"greedy ran out of time after {len(edge_list)} edges. adding random edges."
                )
                remaining_edge_count = num_edges - len(edge_list)

                for j in range(remaining_edge_count):
                    if len(candidate_edges) == 0:
                        logger.info(
                            f"greedy has no candidate edges at step {len(edge_list) + 1}"
                        )
                        break
                    new_edge = edge_selection_rs.choice(list(candidate_edges))
                    edge_list.append(new_edge)
                    candidate_edges = get_candidate_edges(
                        edge_list, graph, max_level_for_pruning
                    )

                return edge_list

            else:

                weight_list, prob_list, _ = evaluate_edge_list(
                    edge_list + [e], graph, num_leaf_samples, edge_selection_rs
                )
                edge_value = np.mean(weight_list)

                if edge_value > max_val:
                    best_edge = e
                    max_val = edge_value

        candidate_edges.remove(best_edge)
        edge_list.append(best_edge)
    return edge_list
