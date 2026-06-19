"""
Consumes graph and student pathways to generate classifications based on the length/pathway
See https://dl.acm.org/doi/10.1145/3636555.3636929
Both individual and global classifications are generated.
Global classifications use the entire sample of students/
Individual classifications are based only on the students' own attempts and pathways.
                This only changes the definitions of deadend and incomplete classifications.
                Optimal and suboptimal classifications remain the same for both of them.
"""

import pandas as pd
import multiprocessing as mp
import networkx as nx

CLASS_BEST = "optimal"
CLASS_DEAD_END = "dead_end"
CLASS_SUB_OPTIMAL = "sub_optimal"
CLASS_INCOMPLETE = "incomplete"
MISSING_NODE = "missing_node"
CLASS_TYPE_MAPPING = {
    "studentSessionId": "category",
    "attemptHlc": "category",
    "visitId": "category",
    "MFL_global_class": "category",
    "MFL_individual_class": "category",
}


def get_attempt_class(
    attempt,
    start_state,
    goal_state,
    best_path_len,
    deadend_nodes,
    student_deadend_nodes,
):
    attempt_len = len(attempt) - 1
    attempt_set = set(attempt)  # attempts may include duplicates
    try:
        attempt_set.remove(start_state)
        attempt_set.remove(goal_state)
    except (
        KeyError
    ):  # ignore if goal or start state missing (can happen for incomplete attempts)
        pass
    # if reached goal state and has the optimal length
    if MISSING_NODE in attempt_set:
        # if there are missing steps in the logs, we cannot classify the attempt,
        # return NA for both classifications and a flag for missing steps
        return pd.NA, pd.NA, True

    if attempt[-1] == goal_state:
        if attempt_len == best_path_len:
            return CLASS_BEST, CLASS_BEST, False
        else:  # if the last step is goal state and the length is > optimal length
            return CLASS_SUB_OPTIMAL, CLASS_SUB_OPTIMAL, False
    else:  # incomplete (either deadend or true incomplete)
        # global deadend classification
        if len(attempt_set.intersection(deadend_nodes)) > 0:
            attempt_class_global = CLASS_DEAD_END
        else:
            attempt_class_global = CLASS_INCOMPLETE
        # individual deadend classification
        if len(attempt_set.intersection(student_deadend_nodes)) > 0:
            attempt_class_individual = CLASS_DEAD_END
        else:
            attempt_class_individual = CLASS_INCOMPLETE
        return attempt_class_global, attempt_class_individual, False


def get_student_dead_end_set(student_G, start_state, goal_state):
    """Adds classifications to pathway nodes and edges for individual student graphs,
        also returns dead end node unique to the student

    Args:
        student_G (dict): Graph of the students transformations for a problem
        start_state (str): start state
        goal_state (str): goal state
    """
    dead_end_set = set()
    try:
        for node in student_G.nodes:
            if not nx.has_path(student_G, node, goal_state):
                if node not in [start_state, goal_state]:
                    dead_end_set.add(node)
    except nx.exception.NodeNotFound:
        pass  # if no student completed the problem
    return dead_end_set


def process_problem_classification(
    problem_student_pathways,
    problem_stu_2_G_map,
    start_state,
    goal_state,
):
    """Returns a dataframe with the problems attempt classification and related data

    Args:
        problem_student_pathways (dict): dict mapping student id to their pathways
        problem_stu_2_G_map (dict): dict mapping student id to individual student graph
        start_state (str): start state
        goal_state (str): goal state
    """

    # remove non student ids from the list (metadata keys; kept for backwards compatibility with old code)
    stu_id_list: list[str] = [
        key
        for key in problem_student_pathways.keys()
        if key not in ["start_state", "goal_state", "optimal_paths", "dead_end_nodes"]
    ]
    global_dead_end_set = set(problem_student_pathways.get("dead_end_nodes", []))
    optimal_len = len(problem_student_pathways.get("optimal_paths", [[]])[0]) - 1

    df_list = []
    for stu_id in stu_id_list:
        student_pathway_data: dict = problem_student_pathways[stu_id]
        student_pathways = student_pathway_data["paths"]  # list of attempts
        global_class_list = []
        student_class_list = []
        has_missing_steps_list = []
        # dead_end nodes: global and individual
        student_dead_end_set = get_student_dead_end_set(
            problem_stu_2_G_map[stu_id], start_state, goal_state
        )
        for student_attempt in student_pathways:
            global_class, individual_class, has_missing_steps = get_attempt_class(
                student_attempt,
                start_state,
                goal_state,
                optimal_len,
                global_dead_end_set,
                student_dead_end_set,
            )
            # global classification
            global_class_list.append(global_class)
            # student/individual binned classification
            student_class_list.append(individual_class)
            has_missing_steps_list.append(has_missing_steps)

        student_class_dict = {
            "studentSessionId": [stu_id] * len(student_pathways),
            "attemptHlc": student_pathway_data["attemptHlc"],
            "visitId": student_pathway_data["visitId"],
            "MFL_global_class": global_class_list,
            "MFL_individual_class": student_class_list,
            "missing_steps_in_logs": has_missing_steps_list,
        }
        df = pd.DataFrame.from_dict(student_class_dict).astype(CLASS_TYPE_MAPPING)

        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)
