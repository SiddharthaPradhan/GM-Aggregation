from typing import Any
import networkx as nx
from networkx import node_link_data
import pandas as pd
import numpy as np
import logging
from .utils import TXT_SEPARATOR, EventLogTypes, EventMistakeTypes, ACTION_EVENTS

START_STATE = "start_state"
GOAL_STATE = "goal_state"


IGNORE_GRAPH_TYPES = [
    EventLogTypes.VISIT_TASK,
    EventLogTypes.VISIT_GAME_SCREEN,
    EventLogTypes.SHOW_HINT,
    EventLogTypes.HIDE_HINT,
    EventLogTypes.MATH_MISTAKE,
]
AUTO_RESET = "autoReset"
RETRY_RESET = "retryButton"
IGNORE_RESET_REASON_TYPES = [
    RETRY_RESET,  # will already have a row for completed attempt
    AUTO_RESET,  # automatically reset by the system after completion
]
logger = logging.getLogger("GM-Aggregator." + __name__)


def remove_edge_from_graph(
    G: nx.DiGraph, prev_state: str, new_state: str
) -> nx.DiGraph:
    if G.has_edge(prev_state, new_state):
        # only add unique student ID
        attempt_count = G[prev_state][new_state]["attempt_count"]
        attempt_count -= (
            1  # decrease the number of attempts done through this transformation
        )
        if attempt_count == 0:
            G.remove_edge(prev_state, new_state)
            G.remove_node(new_state)
        else:
            G[prev_state][new_state]["attempt_count"] = attempt_count
    return G


def add_edge_to_graph(
    G: nx.DiGraph, prev_state: str, new_state: str, math_trans_type: str = ""
) -> nx.DiGraph:
    if G.has_edge(prev_state, new_state):
        attempt_count = G[prev_state][new_state]["attempt_count"]
        # increments the number of attempts done through this transformation
        attempt_count += 1
        G[prev_state][new_state]["attempt_count"] = attempt_count
    else:
        # only 1 student has made this transformation so far
        G.add_edge(
            prev_state, new_state, attempt_count=1, math_trans_type=math_trans_type
        )
    return G


def add_attempt_to_graphs(global_G, student_G, states, actions):
    for i in range(len(states) - 1):
        prev_state = states[i]
        new_state = states[i + 1]
        math_trans_type = actions[i]
        global_G = add_edge_to_graph(global_G, prev_state, new_state, math_trans_type)
        student_G = add_edge_to_graph(student_G, prev_state, new_state, math_trans_type)
    return global_G, student_G


def make_problem_graph(
    problem_event_df: pd.DataFrame,
    problem_id: str,
    start_state: str,
    goal_state: str,
    store_attempt_meta=False,
) -> tuple[dict | None, dict, dict]:
    """Returns graph JSON, student graphs, and student paths for a problem"""
    # sort by studentSessionId, taskNumber, timestamp
    problem_event_df = problem_event_df.sort_values(
        by=["studentSessionId", "taskNumber", "timestamp"]
    )

    # remove empty visits
    # mask for visits that contains at least one user interaction event
    no_empty_visits = problem_event_df.groupby(
        ["studentSessionId", "taskNumber", "attemptHlc", "visitId"],
        observed=True,
        sort=False,
    )["eventType"].transform(lambda s: s.isin(ACTION_EVENTS).any())
    no_empty_visits = no_empty_visits.astype(bool).fillna(True)
    problem_event_df = problem_event_df[no_empty_visits].reset_index(drop=True)

    problem_event_df = problem_event_df.loc[
        ~problem_event_df["eventType"].isin(IGNORE_GRAPH_TYPES)
    ]
    problem_event_df = problem_event_df.loc[
        ~(
            (problem_event_df["eventType"] == EventLogTypes.RESET_TASK)
            & (problem_event_df["reason"].isin(IGNORE_RESET_REASON_TYPES))
        )
    ]
    problem_event_df.sort_values(
        by=["studentSessionId", "timestamp"], inplace=True, ignore_index=True
    )

    global_G = nx.DiGraph()  # init new global graph
    student_G_dict: dict[str, nx.DiGraph] = {}  # dict for individual student graphs
    student_paths: dict[str, Any] = {}  # init student path JSONstudent_attempts = []
    student_paths[START_STATE] = start_state
    student_paths[GOAL_STATE] = goal_state
    for (studentSessionId, attemptHlc, visit_id), group_df in problem_event_df.groupby(
        by=["studentSessionId", "attemptHlc", "visitId"], observed=True, sort=False
    ):
        if studentSessionId not in student_G_dict:
            student_G_dict[studentSessionId] = (
                nx.DiGraph()
            )  # init new graph for student
        student_G = student_G_dict[studentSessionId]
        # undo and redo events make it impossible to vectorize
        states = [start_state]
        actions = []
        undo_state_stack: list[str] = []
        undo_action_stack: list[str] = []
        attempt_finished = False  # completed or reset
        reached_goal = False  # did student complete the problem?
        # loop over the events in the attempt to reconstruct the path
        for row in group_df.itertuples(index=False):
            if attempt_finished:
                logger.warning(
                    f"Warning: student {studentSessionId} has events after finishing the problem in attemptHlc {attemptHlc}"
                )
                break
            event_type = row.eventType
            if event_type == EventLogTypes.MATH_STEP:
                # todo convert to ascii math
                new_state = row.newState
                states.append(new_state)
                actions.append(row.actionName)
                undo_state_stack.clear()  # clear the undo stack on a new step
                undo_action_stack.clear()  # clear the undo action stack on a new step
            elif event_type == EventLogTypes.SOLVED_TASK:
                # force replace the final state with the goal state to allow commuted answers
                states[-1] = goal_state
                attempt_finished = True
                reached_goal = True
            elif event_type == EventLogTypes.RESET_TASK:
                attempt_finished = True
            elif event_type == EventLogTypes.UNDO_STEP:
                undo_state_stack.append(states.pop())
                undo_action_stack.append(actions.pop())
            elif event_type == EventLogTypes.REDO_STEP:
                states.append(undo_state_stack.pop())
                actions.append(undo_action_stack.pop())
        global_G, student_G = add_attempt_to_graphs(
            global_G, student_G, states, actions
        )
        student_G_dict[studentSessionId] = student_G

        if not student_paths.get(studentSessionId, False):
            path_dict = {"paths": [states], "is_completed": [reached_goal]}
            if store_attempt_meta:
                path_dict["attemptHlc"] = [attemptHlc]
                path_dict["visitId"] = [visit_id]

            student_paths[studentSessionId] = path_dict
        else:
            student_paths[studentSessionId]["paths"].append(states)
            student_paths[studentSessionId]["is_completed"].append(reached_goal)
            if store_attempt_meta:
                student_paths[studentSessionId]["attemptHlc"].append(attemptHlc)
                student_paths[studentSessionId]["visitId"].append(visit_id)

    try:
        optimal_paths = list(
            nx.all_shortest_paths(global_G, start_state, goal_state, method="dijkstra")
        )
        for path in optimal_paths:
            for i in range(0, len(path) - 1):
                global_G[path[i]][path[i + 1]]["is_best_path"] = True
            path = path[1:-1]  # remove start and goal state
            for node in path:
                global_G.nodes[node]["in_best_path"] = True
        student_paths["optimal_paths"] = optimal_paths
        dead_end_nodes = []
        for node in global_G.nodes:
            if not nx.has_path(global_G, node, goal_state):
                global_G.nodes[node]["is_dead_end"] = True
                dead_end_nodes.append(node)
        student_paths["dead_end_nodes"] = dead_end_nodes

    except nx.exception.NetworkXNoPath:
        logger.info(f"Problem {problem_id} has not been solved so far.")
    global_graph_json = node_link_data(global_G, edges="links")
    return global_graph_json, student_G_dict, student_paths
