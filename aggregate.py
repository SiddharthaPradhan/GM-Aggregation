import pandas as pd
import numpy as np
import multiprocessing as mp
import logging
from tqdm import tqdm
from utils import (
    save_df,
    write_to_study_meta_text,
    EventLogTypes,
    EventMistakeTypes,
    OUTPUT_TYPE,
    EVENT_COLS_TO_KEEP,
    ACTION_EVENTS,
)

logger = logging.getLogger(__name__)


def aggregate_and_save(
    event_log_df: pd.DataFrame,
    output_dir: str,
    output_type: OUTPUT_TYPE,
    n_jobs: int = -1,
):
    """Aggregates the event log data to the attempt, student-problem, student, problem and overall levels.

    Args:
        event_log_df (pd.DataFrame): DataFrame containing preprocessed event log data.
        output_type (OUTPUT_TYPE, optional): Whether to save as csv or in sqlite. Defaults to "sqlite".
    """
    logger.info("Starting Aggregation..")
    attempt_level_df, student_problem_df = aggregate_event_log(
        event_log_df, n_jobs=n_jobs
    )
    logger.debug("Finished Attempt and Student-Problem Level Aggregation..")
    save_df(attempt_level_df, "attempt_level", output_dir, output_type=output_type)
    save_df(student_problem_df, "student_problem", output_dir, output_type=output_type)

    student_df = _aggregate_to_student_level(student_problem_df)
    logger.debug("Finished Student Level Aggregation..")
    problem_df = _aggregate_to_problem_level(student_problem_df)
    logger.debug("Finished Problem Level Aggregation..")
    assignment_df = _aggregate_to_assignment_level(student_df)

    save_df(student_df, "student_level", output_dir, output_type=output_type)
    save_df(problem_df, "problem_level", output_dir, output_type=output_type)
    write_to_study_meta_text(output_dir, assignment_df.to_string())


def aggregate_event_log(event_log_df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """Aggregate event log data to attempt level and .

    Args:
        event_log_df (pd.DataFrame): DataFrame containing preprocessed event log data.
        n_jobs (int, optional): Number of parallel jobs to use. Defaults to -1 (use all available cores).
    Returns:
        pd.DataFrame: DataFrame aggregated to attempt level.
    """
    event_log_df = event_log_df[EVENT_COLS_TO_KEEP]  # keep only relevant columns

    # all task numbers
    task_numbers = event_log_df["taskNumber"].unique()
    task_numbers = task_numbers[~np.isnan(task_numbers)]
    mp_jobs = []
    for task_number in task_numbers:
        mp_jobs.append(
            event_log_df.loc[event_log_df["taskNumber"] == task_number].copy()
        )
    if n_jobs == -1:
        num_processes = None  # Use all available cores
    else:
        num_processes = min(n_jobs, len(mp_jobs))
    df_list_attempt = []
    df_list_student_problem = []
    with (
        mp.get_context("spawn").Pool(processes=num_processes) as pool,
        tqdm(
            total=len(mp_jobs), desc="Aggregating Event Logs To Attempt Level"
        ) as pbar,
    ):
        for result in pool.imap_unordered(_aggregate_single_problem_logs, mp_jobs):
            pbar.update()
            pbar.refresh()
            partial_attempt_level_df, partial_student_problem_df = result
            df_list_attempt.append(partial_attempt_level_df)
            df_list_student_problem.append(partial_student_problem_df)
    attempt_level_df = pd.concat(df_list_attempt, ignore_index=True)
    student_problem_df = pd.concat(df_list_student_problem, ignore_index=True)
    attempt_level_df = attempt_level_df.sort_values(
        by=["studentRef", "taskNumber", "attemptNumber"]
    ).reset_index(drop=True)
    student_problem_df = student_problem_df.sort_values(
        by=["studentRef", "taskNumber"]
    ).reset_index(drop=True)
    return attempt_level_df, student_problem_df


def _aggregate_to_student_level(student_problem_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate student-problem level to student level.
    Args:
        student_problem_df (pd.DataFrame): DataFrame containing student-problem level data.
    Returns:
        pd.DataFrame: DataFrame aggregated to student level.
    """
    student_df = (
        student_problem_df.groupby("studentRef")
        .agg(
            total_problems=("taskNumber", "nunique"),
            total_completed_problems=("problem_completed", "sum"),
            total_attempts=("num_attempts", "sum"),
            total_replays=("num_replays", "sum"),
            total_completed_attempts=("num_completed_attempts", "sum"),
            total_errors=("num_errors", "sum"),
            total_keypad_errors=("num_keypad_errors", "sum"),
            total_shaking_errors=("num_shaking_errors", "sum"),
            total_snapping_errors=("num_snapping_errors", "sum"),
            total_hints=("num_hints", "sum"),
            total_time_spent=("total_time_spent", "sum"),
            total_one_star=("num_one_star", "sum"),
            total_two_star=("num_two_star", "sum"),
            total_three_star=("num_three_star", "sum"),
            total_incomplete=("num_incomplete", "sum"),
            avg_stars=("avg_stars", "mean"),
        )
        .reset_index()
    )

    student_df["total_time_spent_mins"] = student_df["total_time_spent"] / 60
    return student_df


def _aggregate_to_problem_level(student_problem_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate student-problem level to problem level.
    Args:
        student_problem_df (pd.DataFrame): DataFrame containing student-problem level data.
    Returns:
        pd.DataFrame: DataFrame aggregated to problem level.
    """
    problem_df = (
        student_problem_df.groupby("taskNumber")
        .agg(
            total_students=("studentRef", "nunique"),
            avg_time_spent=("total_time_spent", "mean"),
            avg_attempts=("num_attempts", "mean"),
            avg_replays=("num_replays", "mean"),
            avg_completed_attempts=("num_completed_attempts", "mean"),
            avg_errors=("num_errors", "mean"),
            avg_keypad_errors=("num_keypad_errors", "mean"),
            avg_shaking_errors=("num_shaking_errors", "mean"),
            avg_snapping_errors=("num_snapping_errors", "mean"),
            avg_hints=("num_hints", "mean"),
            avg_one_star=("num_one_star", "mean"),
            avg_two_star=("num_two_star", "mean"),
            avg_three_star=("num_three_star", "mean"),
            avg_incomplete=("num_incomplete", "mean"),
            avg_stars=("avg_stars", "mean"),
        )
        .reset_index()
    )
    problem_df["avg_time_spent_mins"] = problem_df["avg_time_spent"] / 60
    return problem_df


def _aggregate_to_assignment_level(student_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate student level to assignment level.
    Args:
        student_df (pd.DataFrame): DataFrame containing student level data.
    Returns:
        pd.DataFrame: DataFrame aggregated to assignment level (using pd.DataFrame.describe).
    """
    return student_df.describe().drop("count")


def _aggregate_single_student_problem(problem_attempt_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate attempt-level to student-problem level for a single problem.
    Args:
        problem_attempt_df (pd.DataFrame): DataFrame containing attempt-level data for a single problem.
    Returns:
        pd.DataFrame: DataFrame aggregated to student-problem level for a single problem
    """
    problem_attempt_df["stars_1"] = problem_attempt_df["stars"] == 1
    problem_attempt_df["stars_2"] = problem_attempt_df["stars"] == 2
    problem_attempt_df["stars_3"] = problem_attempt_df["stars"] == 3
    problem_attempt_df["stars_NA"] = problem_attempt_df["stars"].isna()
    student_problem_df = (
        problem_attempt_df.groupby(["studentRef", "taskNumber"])
        .agg(
            num_attempts=("attemptNumber", "max"),
            num_replays=("replay_attempt", "sum"),
            problem_completed=("completed_dur_attempt", "any"),
            num_completed_attempts=("completed_dur_attempt", "sum"),
            num_errors=("num_errors", "sum"),
            num_keypad_errors=("num_keypad_errors", "sum"),
            num_shaking_errors=("num_shaking_errors", "sum"),
            num_snapping_errors=("num_snapping_errors", "sum"),
            num_hints=("num_hints", "sum"),
            total_time_spent=("total_time", "sum"),
            avg_time_spent=("total_time", "mean"),
            avg_pause_time=("pause_time", "mean"),
            avg_pause_time_ratio=("pause_time_ratio", "mean"),
            num_three_star=("stars_3", "sum"),
            num_two_star=("stars_2", "sum"),
            num_one_star=("stars_1", "sum"),
            num_incomplete=("stars_NA", "sum"),
            avg_stars=("stars", "mean"),
        )
        .reset_index()
    )
    student_problem_df["num_completed_attempts"] = (
        student_problem_df["num_completed_attempts"].fillna(0).astype(int)
    )
    student_problem_df["num_replays"] = (
        student_problem_df["num_replays"].fillna(0).astype(int)
    )
    return student_problem_df


def _aggregate_single_problem_logs(
    problem_event_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate event log data for a single problem.

    Args:
        problem_event_df (pd.DataFrame): DataFrame containing event log data for a single problem.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrame aggregated to attempt level and student-problem level for a single problem.

    """
    # remove redundant row
    problem_event_df = problem_event_df.loc[
        ~problem_event_df["eventType"].isin(["visitGameScreen"])
    ].copy()
    # sort by studentRef, taskNumber, timestamp
    problem_event_df = problem_event_df.sort_values(
        by=["studentRef", "taskNumber", "timestamp"]
    )

    # fix reset task belonging to the previous attempt
    # doing this will ensure time after completion or last action is not counted towards total_time
    problem_event_df["next_attemptHlc"] = problem_event_df.groupby(
        ["studentRef", "taskNumber"]
    )["attemptHlc"].shift(-1)
    # increment attemptNumber and set attemptHlc to next_attemptHlc for resetTask events
    problem_event_df.loc[
        problem_event_df["eventType"] == "resetTask", "attemptNumber"
    ] += 1
    problem_event_df.loc[problem_event_df["eventType"] == "resetTask", "attemptHlc"] = (
        problem_event_df.loc[
            problem_event_df["eventType"] == "resetTask", "next_attemptHlc"
        ]
    )

    # group by student-task-attempts (unique attempts)
    student_attempt_group = problem_event_df.groupby(
        ["studentRef", "taskNumber", "attemptNumber"]
    )

    problem_event_df["next_timestamp"] = student_attempt_group["timestamp"].shift(-1)
    # record cases where resetTask has timestamp greater than next event timestamp
    # this indicates an error in timestamp recording
    reset_time_error_flag = (problem_event_df["eventType"] == "resetTask") & (
        problem_event_df["timestamp"] > problem_event_df["next_timestamp"]
    )
    problem_event_df["reset_time_error"] = reset_time_error_flag
    # update timestamp to next timestamp for resetTask with time error
    problem_event_df.loc[reset_time_error_flag, "timestamp"] = problem_event_df.loc[
        reset_time_error_flag, "next_timestamp"
    ]

    # add next event column
    problem_event_df["next_event"] = student_attempt_group["eventType"].shift(-1)
    # create mask to drop double visitTask that are consecutive
    drop_mask = (problem_event_df["eventType"] == EventLogTypes.VISIT_TASK) & (
        problem_event_df["next_event"] == EventLogTypes.VISIT_TASK
    )
    problem_event_df = problem_event_df[~drop_mask].copy()

    # add previous timestamp column
    problem_event_df["previous_timestamp"] = student_attempt_group["timestamp"].shift(1)

    # compute delta time (time difference between current and previous event)
    problem_event_df["delta_time"] = (
        problem_event_df["timestamp"] - problem_event_df["previous_timestamp"]
    )

    # zip delta time with whether event is an action event
    # used when computing pause time
    #   (this avoids having to use the slower apply method later)
    problem_event_df["delta_time_tuple"] = list(
        zip(
            problem_event_df["delta_time"],
            problem_event_df["eventType"].isin(ACTION_EVENTS),
        )
    )
    # regroup after adding new columns and filtering
    student_attempt_group = problem_event_df.groupby(
        ["studentRef", "taskNumber", "attemptNumber"]
    )

    # attempt aggregation
    attempt_agg_df = student_attempt_group.agg(
        start_time=("timestamp", "first"),
        # end time needs to ignore time after solvingTask (e.g. reset)
        end_time=("timestamp", "last"),
        total_time=("delta_time", _sum_delta_times_under_5min),
        pause_time=("delta_time_tuple", _get_pause_time),
        num_steps=("eventType", lambda x: (x == EventLogTypes.MATH_STEP).sum()),
        num_errors=("eventType", lambda x: (x == EventLogTypes.MATH_MISTAKE).sum()),
        # no goal state checking, based only on event logs
        completed_dur_attempt=(
            "eventType",
            lambda x: (x == EventLogTypes.SOLVED_TASK).any(),
        ),
        num_visits=("eventType", lambda x: (x == EventLogTypes.VISIT_TASK).sum()),
        validity_first_step=("eventType", _get_validity_first_step),
        num_hints=("eventType", lambda x: (x == EventLogTypes.SHOW_HINT).sum()),
        made_action=("eventType", lambda x: x.isin(ACTION_EVENTS).any()),
        num_keypad_errors=("mistake", lambda x: (x == EventMistakeTypes.KEYPAD).sum()),
        num_shaking_errors=("mistake", lambda x: (x == EventMistakeTypes.TAP).sum()),
        num_snapping_errors=("mistake", lambda x: (x == EventMistakeTypes.DRAG).sum()),
        stars=("stars", "max"),
        hlc_changed_dur_attempt=("attemptHlc", lambda x: x.nunique() > 1),
        reset_time_error=("reset_time_error", "any"),
    ).reset_index()

    attempt_agg_df["pause_time_ratio"] = (
        attempt_agg_df["pause_time"] / attempt_agg_df["total_time"]
    )
    attempt_agg_df["replay_attempt"] = attempt_agg_df.groupby(
        ["studentRef", "taskNumber"]
    )["completed_dur_attempt"].transform(_get_replay_attempt)

    # set dtypes
    attempt_agg_df = _set_attempt_dtypes(attempt_agg_df)
    student_problem_df = _aggregate_single_student_problem(attempt_agg_df)
    return attempt_agg_df, student_problem_df


def _set_attempt_dtypes(attempt_agg_df: pd.DataFrame) -> pd.DataFrame:
    return attempt_agg_df.astype(
        {
            "studentRef": str,
            "taskNumber": int,
            "attemptNumber": int,
            "total_time": float,
            "pause_time": float,
            "pause_time_ratio": float,
            "replay_attempt": bool,
            "num_steps": int,
            "num_errors": int,
            "num_keypad_errors": int,
            "num_shaking_errors": int,
            "num_snapping_errors": int,
            "num_hints": int,
            "completed_dur_attempt": bool,
            # "stars": int,
            "num_visits": int,
            "validity_first_step": bool,
            "made_action": bool,
            "hlc_changed_dur_attempt": bool,
            "reset_time_error": bool,
        }
    )


def _get_pause_time(col: pd.Series) -> float:
    col = col.apply(pd.Series)  # expand tuples
    action_times: pd.Series[pd.Timedelta] = col.loc[col[1] == True, 0]

    if len(action_times) == 0:  # no actions for this attempt
        return None
    else:
        return action_times.iloc[0].total_seconds()


def _sum_delta_times_under_5min(delta_times: pd.Series) -> float:
    """Sum delta times that are less than 5 minutes."""
    # Filter delta times less than 5 minutes (300 seconds)
    valid_times = delta_times[delta_times < pd.Timedelta(minutes=5)]

    if len(valid_times) == 0:
        return 0.0
    else:
        return valid_times.sum().total_seconds()


def _get_validity_first_step(col: pd.Series) -> bool:
    is_mistake = col == EventLogTypes.MATH_MISTAKE
    is_math_step = col == EventLogTypes.MATH_STEP
    if not is_math_step.any():  # if no math steps
        return None
    elif not is_mistake.any():  # if no math mistakes and at least one math step
        return True
    else:  # at least one math step and mistake
        first_mistake_index = is_mistake.idxmax()
        first_math_step_index = is_math_step.idxmax()
        return first_math_step_index < first_mistake_index


def _get_replay_attempt(completed_dur_attempt: pd.Series) -> pd.Series:
    if not completed_dur_attempt.any():
        # no replays attempts if there is no completed attempt
        return pd.Series([False] * len(completed_dur_attempt))
    else:
        # any attempt after the first completed attempt is a replay attempt
        first_completed_index = completed_dur_attempt.idxmax()
        return completed_dur_attempt.index > first_completed_index
