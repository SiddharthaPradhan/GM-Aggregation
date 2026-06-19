from typing import cast
from typing import Callable, TypedDict
import pandas as pd
import numpy as np
import multiprocessing as mp
import logging
import gc
import tempfile
import os
from tqdm import tqdm
from .utils import (
    save_df,
    write_to_study_meta_text,
    EventLogTypes,
    EventMistakeTypes,
    OUTPUT_TYPE,
    ACTION_EVENTS,
    ResetReasons,
)
import networkx as nx
from .generate_graph import make_problem_graph
from .generate_classifications import process_problem_classification
from .preprocess import TaskSplitEventLog

logger = logging.getLogger("GM-Aggregator." + __name__)


class ProgressEvent(TypedDict, total=False):
    stage: str
    completed: int
    total: int
    message: str


ProgressCallback = Callable[[ProgressEvent], None]


def _stream_concat_csv_files(
    file_paths: list[str],
    dtype_map: dict | None = None,
    chunk_size: int = 500000,
    sort_by: list[str] | None = None,
) -> pd.DataFrame:
    """Read CSV files in chunks and concatenate without holding all data in memory.

    Processes each file in chunks to minimize peak memory usage. If sort_by is provided,
    uses in-place sorting to avoid creating temporary copies during sort operations.
    """
    all_chunks = []

    for file_path in file_paths:
        if os.path.getsize(file_path) == 0:
            continue

        for chunk in pd.read_csv(
            file_path,
            dtype=dtype_map,
            chunksize=chunk_size,
        ):
            all_chunks.append(chunk)
            if len(all_chunks) >= 2:
                intermediate_df = pd.concat(all_chunks, ignore_index=True)
                del all_chunks
                all_chunks = [intermediate_df]
                gc.collect()

    if not all_chunks:
        return pd.DataFrame()

    result_df = pd.concat(all_chunks, ignore_index=True)

    # Use in-place sort to avoid creating a copy—critical on memory-constrained systems
    if sort_by:
        result_df.sort_values(by=sort_by, inplace=True, ignore_index=True)

    gc.collect()
    return result_df


ATTEMPT_TYPE_MAPPING = {
    "studentSessionId": "category",
    "taskNumber": "uint16[pyarrow]",
    "studentRef": "category",
    "start_time": "datetime64[ns]",
    "end_time": "datetime64[ns]",
    "total_time": "float[pyarrow]",
    "pause_time": "float[pyarrow]",
    "num_steps": "uint16[pyarrow]",
    "num_errors": "uint16[pyarrow]",
    "completed_dur_attempt": "bool[pyarrow]",
    "num_visits": "uint16[pyarrow]",
    "validity_first_step": "bool[pyarrow]",
    "num_hints": "uint16[pyarrow]",
    "num_keypad_errors": "uint16[pyarrow]",
    "num_shaking_errors": "uint16[pyarrow]",
    "num_snapping_errors": "uint16[pyarrow]",
    "stars": "uint16[pyarrow]",
    "clicked_reset_button": "bool[pyarrow]",
    "clicked_retry_button": "bool[pyarrow]",
    "pause_time_ratio": "float[pyarrow]",
    "replay_attempt": "bool[pyarrow]",
    "attempt_number": "uint16[pyarrow]",
    "attempt_visit_number": "uint16[pyarrow]",
}

STUDENT_PROBLEM_TYPE_MAPPING = {
    "studentSessionId": "category",
    "taskNumber": "uint16[pyarrow]",
    "studentRef": "category",
    "num_attempts": "uint16[pyarrow]",
    "num_replays": "uint16[pyarrow]",
    "num_clicked_retry_button": "uint16[pyarrow]",
    "num_clicked_reset_button": "uint16[pyarrow]",
    "problem_completed": "bool[pyarrow]",
    "num_completed_attempts": "uint16[pyarrow]",
    "num_errors": "uint16[pyarrow]",
    "num_keypad_errors": "uint16[pyarrow]",
    "num_shaking_errors": "uint16[pyarrow]",
    "num_snapping_errors": "uint16[pyarrow]",
    "num_hints": "uint16[pyarrow]",
    "total_time_spent": "float[pyarrow]",
    "avg_time_spent": "float[pyarrow]",
    "avg_pause_time": "float[pyarrow]",
    "avg_pause_time_ratio": "float[pyarrow]",
    "num_three_star": "uint16[pyarrow]",
    "num_two_star": "uint16[pyarrow]",
    "num_one_star": "uint16[pyarrow]",
    "num_incomplete": "uint16[pyarrow]",
    "avg_stars": "float[pyarrow]",
}

PROBLEM_TYPE_MAPPING = {
    "taskNumber": "uint16[pyarrow]",
    "total_students": "uint16[pyarrow]",
    "avg_time_spent": "float[pyarrow]",
    "avg_attempts": "float[pyarrow]",
    "avg_replays": "float[pyarrow]",
    "avg_completed_attempts": "float[pyarrow]",
    "avg_errors": "float[pyarrow]",
    "avg_keypad_errors": "float[pyarrow]",
    "avg_shaking_errors": "float[pyarrow]",
    "avg_snapping_errors": "float[pyarrow]",
    "avg_hints": "float[pyarrow]",
    "avg_one_star": "float[pyarrow]",
    "avg_two_star": "float[pyarrow]",
    "avg_three_star": "float[pyarrow]",
    "avg_incomplete": "float[pyarrow]",
    "avg_stars": "float[pyarrow]",
    "avg_time_spent_mins": "float[pyarrow]",
    "avg_click_reset": "float[pyarrow]",
    "avg_click_retry": "float[pyarrow]",
}

STUDENT_TYPE_MAPPING = {
    "studentSessionId": "category",
    "studentRef": "category",
    "total_problems": "uint16[pyarrow]",
    "total_completed_problems": "uint16[pyarrow]",
    "total_attempts": "uint16[pyarrow]",
    "total_replays": "uint16[pyarrow]",
    "total_completed_attempts": "uint16[pyarrow]",
    "total_errors": "uint16[pyarrow]",
    "total_keypad_errors": "uint16[pyarrow]",
    "total_shaking_errors": "uint16[pyarrow]",
    "total_snapping_errors": "uint16[pyarrow]",
    "total_hints": "uint16[pyarrow]",
    "total_time_spent": "float[pyarrow]",
    "total_one_star": "uint16[pyarrow]",
    "total_two_star": "uint16[pyarrow]",
    "total_three_star": "uint16[pyarrow]",
    "total_incomplete": "uint16[pyarrow]",
    "avg_stars": "float[pyarrow]",
    "total_click_reset": "uint16[pyarrow]",
    "total_click_retry": "uint16[pyarrow]",
    "avg_click_reset": "float[pyarrow]",
    "avg_click_retry": "float[pyarrow]",
    "total_time_spent_mins": "float[pyarrow]",
}


def aggregate_and_save(
    event_log_df: pd.DataFrame,
    task_metadata_df: pd.DataFrame,
    output_dir: str,
    output_type: OUTPUT_TYPE,
    n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
    dask_client=None,
):
    """Aggregates the event log data to the attempt, student-problem, student, problem and overall levels.

    Args:
        event_log_df (pd.DataFrame): DataFrame containing preprocessed event log data.
        output_type (OUTPUT_TYPE, optional): Whether to save as csv or in sqlite. Defaults to "sqlite".
    """
    logger.info("Starting Aggregation..")
    attempt_level_df, student_problem_df = aggregate_event_log(
        event_log_df,
        task_metadata_df,
        n_jobs=n_jobs,
        progress_callback=progress_callback,
        dask_client=dask_client,
    )
    logger.debug("Finished Attempt and Student-Problem Level Aggregation..")

    # Sort attempt_level and student_problem before saving
    attempt_level_df = attempt_level_df.astype(ATTEMPT_TYPE_MAPPING)
    attempt_level_df.sort_values(
        by=["studentSessionId", "taskNumber", "attemptHlc", "visitId"],
        inplace=True,
        ignore_index=True,
    )
    save_df(attempt_level_df, "attempt_level", output_dir, output_type=output_type)
    del attempt_level_df
    gc.collect()

    student_problem_df = student_problem_df.astype(STUDENT_PROBLEM_TYPE_MAPPING)
    student_problem_df.sort_values(
        by=["studentSessionId", "taskNumber"],
        inplace=True,
        ignore_index=True,
    )
    save_df(student_problem_df, "student_problem", output_dir, output_type=output_type)

    student_df = _aggregate_to_student_level(student_problem_df)
    student_df.sort_values(by="studentSessionId", inplace=True, ignore_index=True)
    logger.debug("Finished Student Level Aggregation..")

    problem_df = _aggregate_to_problem_level(student_problem_df)
    problem_df.sort_values(by="taskNumber", inplace=True, ignore_index=True)
    save_df(problem_df, "problem_level", output_dir, output_type=output_type)
    del problem_df
    gc.collect()
    logger.debug("Finished Problem Level Aggregation..")
    assignment_df = _aggregate_to_assignment_level(student_df)

    save_df(student_df, "student_level", output_dir, output_type=output_type)
    write_to_study_meta_text(output_dir, assignment_df.to_string())


def get_start_goal_state(
    task_metadata_df: pd.DataFrame, task_number: int
) -> tuple[str, str]:
    """Get the start and goal states for a given task number from the task metadata dataframe."""
    task_meta = task_metadata_df.loc[task_metadata_df["taskNumber"] == task_number]
    if len(task_meta) == 0:
        raise ValueError(f"No metadata found for task number {task_number}")
    elif len(task_meta) > 1:
        raise ValueError(
            f"Multiple metadata entries found for task number {task_number}"
        )
    start_state: str = task_meta["startState"].iloc[0]
    goal_state: str = task_meta["goalState"].iloc[0]
    return start_state, goal_state


def aggregate_event_log(
    event_log_df: pd.DataFrame,
    task_metadata_df: pd.DataFrame,
    n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
    dask_client=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate event log data to attempt level and student-problem level.

    Args:
        event_log_df (pd.DataFrame): DataFrame containing preprocessed event log data.
        task_metadata_df (pd.DataFrame): DataFrame containing task metadata.
        n_jobs (int, optional): Number of parallel jobs to use. Defaults to -1 (use all available cores).
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames aggregated to attempt level and student-problem level.
    """

    # all task numbers — handle both a plain pandas DataFrame and a
    # TaskSplitEventLog (per-task files already split out on disk)
    _is_split = isinstance(event_log_df, TaskSplitEventLog)

    if _is_split:
        task_numbers = list(event_log_df.task_files.keys())
    else:
        task_numbers = event_log_df["taskNumber"].unique()
        task_numbers = task_numbers[~np.isnan(task_numbers)]
    total_tasks = len(task_numbers)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "aggregation",
                "completed": 0,
                "total": total_tasks,
                "message": "Starting aggregation",
            }
        )

    df_list_attempt = []
    df_list_student_problem = []
    attempt_tmp_dir = None
    student_problem_tmp_dir = None

    if dask_client is not None:
        from dask.distributed import as_completed as dask_as_completed

        attempt_tmp_dir = tempfile.mkdtemp(prefix="agg-attempt-")
        student_problem_tmp_dir = tempfile.mkdtemp(prefix="agg-student-problem-")
        attempt_file_counter = [0]
        student_problem_file_counter = [0]

        # Lazy submission: only keep a small window of serialised task DataFrames
        # in the scheduler at once so we never hold all N copies in memory.
        n_concurrent = len(dask_client.nthreads()) * 2
        task_iter = iter(task_numbers)

        def _submit_next():
            try:
                tn = next(task_iter)
                # TaskSplitEventLog: read this task's already-split small
                # file directly (fast, no full-file rescan). Plain
                # DataFrame: standard loc filter.
                if _is_split:
                    task_df = event_log_df.read_task(tn)
                else:
                    task_df = event_log_df.loc[event_log_df["taskNumber"] == tn].copy()
                job = (tn, *get_start_goal_state(task_metadata_df, tn), task_df)
                return dask_client.submit(_handle_single_problem_logs, job, pure=False)
            except StopIteration:
                return None

        initial = [
            f
            for f in (_submit_next() for _ in range(min(n_concurrent, total_tasks)))
            if f is not None
        ]
        ac = dask_as_completed(initial)

        with tqdm(
            total=total_tasks,
            desc="Aggregating Event Logs To Attempt Level",
            unit="problem",
            disable=None,
        ) as pbar:
            for n_done, future in enumerate(ac, start=1):
                partial_attempt_level_df, partial_student_problem_df = future.result()
                attempt_file_path = os.path.join(
                    attempt_tmp_dir, f"attempt_{attempt_file_counter[0]}.csv"
                )
                student_problem_file_path = os.path.join(
                    student_problem_tmp_dir,
                    f"student_problem_{student_problem_file_counter[0]}.csv",
                )
                partial_attempt_level_df.to_csv(attempt_file_path, index=False)
                partial_student_problem_df.to_csv(
                    student_problem_file_path, index=False
                )
                attempt_file_counter[0] += 1
                student_problem_file_counter[0] += 1
                del partial_attempt_level_df, partial_student_problem_df
                gc.collect()
                pbar.update()
                pbar.refresh()
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "aggregation",
                            "completed": n_done,
                            "total": total_tasks,
                        }
                    )
                next_f = _submit_next()
                if next_f is not None:
                    ac.add(next_f)

        # Consume any remaining futures and wait for all tasks to complete
        remaining_futures = list(ac)
        if remaining_futures:
            dask_client.gather(remaining_futures)

        # Wait for all workers to become idle before returning
        dask_client.wait_for_workers(len(dask_client.nthreads()))

        if _is_split:
            event_log_df.cleanup()
    else:
        mp_jobs = []
        for task_number in task_numbers:
            mp_jobs.append(
                (
                    task_number,
                    *get_start_goal_state(task_metadata_df, task_number),
                    event_log_df.loc[event_log_df["taskNumber"] == task_number].copy(),
                )
            )
        num_processes = None if n_jobs == -1 else min(n_jobs, len(mp_jobs))
        with (
            mp.get_context("spawn").Pool(processes=num_processes) as pool,
            tqdm(
                total=len(mp_jobs),
                desc="Aggregating Event Logs To Attempt Level",
                unit="problem",
                disable=None,
            ) as pbar,
        ):
            for completed, result in enumerate(
                pool.imap_unordered(_handle_single_problem_logs, mp_jobs), start=1
            ):
                pbar.update()
                pbar.refresh()
                partial_attempt_level_df, partial_student_problem_df = result
                df_list_attempt.append(partial_attempt_level_df)
                df_list_student_problem.append(partial_student_problem_df)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "aggregation",
                            "completed": completed,
                            "total": total_tasks,
                        }
                    )

    try:
        if attempt_tmp_dir is not None and student_problem_tmp_dir is not None:
            attempt_file_paths = sorted(
                [
                    os.path.join(attempt_tmp_dir, f)
                    for f in os.listdir(attempt_tmp_dir)
                    if f.endswith(".csv")
                ]
            )
            student_problem_file_paths = sorted(
                [
                    os.path.join(student_problem_tmp_dir, f)
                    for f in os.listdir(student_problem_tmp_dir)
                    if f.endswith(".csv")
                ]
            )
            attempt_level_df = _stream_concat_csv_files(
                attempt_file_paths,
                sort_by=["studentSessionId", "taskNumber", "start_time"],
            )
            student_problem_df = _stream_concat_csv_files(
                student_problem_file_paths, sort_by=["studentSessionId", "taskNumber"]
            )
        else:
            attempt_level_df = pd.concat(df_list_attempt, ignore_index=True)
            student_problem_df = pd.concat(df_list_student_problem, ignore_index=True)
            attempt_level_df.sort_values(
                by=["studentSessionId", "taskNumber", "start_time"],
                inplace=True,
                ignore_index=True,
            )
            student_problem_df.sort_values(
                by=["studentSessionId", "taskNumber"], inplace=True, ignore_index=True
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "aggregation",
                    "completed": total_tasks,
                    "total": total_tasks,
                    "message": "Finished aggregation",
                }
            )
        return attempt_level_df, student_problem_df
    finally:
        import shutil

        if attempt_tmp_dir and os.path.exists(attempt_tmp_dir):
            shutil.rmtree(attempt_tmp_dir, ignore_errors=True)
        if student_problem_tmp_dir and os.path.exists(student_problem_tmp_dir):
            shutil.rmtree(student_problem_tmp_dir, ignore_errors=True)


def _aggregate_to_student_level(student_problem_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate student-problem level to student level.
    Args:
        student_problem_df (pd.DataFrame): DataFrame containing student-problem level data.
    Returns:
        pd.DataFrame: DataFrame aggregated to student level.
    """
    student_df = (
        student_problem_df.groupby(["studentSessionId"], observed=True, sort=False)
        .agg(
            studentRef=("studentRef", "first"),
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
            total_click_reset=("num_clicked_reset_button", "sum"),
            total_click_retry=("num_clicked_retry_button", "sum"),
            avg_click_reset=("num_clicked_reset_button", "mean"),
            avg_click_retry=("num_clicked_retry_button", "mean"),
            avg_stars=("avg_stars", "mean"),
        )
        .reset_index()
    )

    student_df["total_time_spent_mins"] = student_df["total_time_spent"] / 60
    return student_df.astype(STUDENT_TYPE_MAPPING)


def _aggregate_to_problem_level(student_problem_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate student-problem level to problem level.
    Args:
        student_problem_df (pd.DataFrame): DataFrame containing student-problem level data.
    Returns:
        pd.DataFrame: DataFrame aggregated to problem level.
    """
    problem_df = (
        student_problem_df.groupby("taskNumber", observed=True, sort=False)
        .agg(
            total_students=("studentSessionId", "nunique"),
            total_completed=("problem_completed", "sum"),
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
            avg_click_reset=("num_clicked_reset_button", "mean"),
            avg_click_retry=("num_clicked_retry_button", "mean"),
        )
        .reset_index()
    )
    problem_df["avg_time_spent_mins"] = problem_df["avg_time_spent"] / 60
    return problem_df.astype(PROBLEM_TYPE_MAPPING)


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
        problem_attempt_df.groupby(
            ["studentSessionId", "taskNumber"], observed=True, sort=False
        )
        .agg(
            studentRef=("studentRef", "first"),
            num_attempts=("attempt_number", "max"),
            num_replays=("replay_attempt", "sum"),
            num_clicked_retry_button=("clicked_retry_button", "sum"),
            num_clicked_reset_button=("clicked_reset_button", "sum"),
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
    return _set_student_problem_dtypes(student_problem_df)


def _set_student_problem_dtypes(student_problem_df: pd.DataFrame) -> pd.DataFrame:
    """Set the appropriate dtypes for the student-problem DataFrame."""
    return student_problem_df.astype(STUDENT_PROBLEM_TYPE_MAPPING)


# after discussion with David (on June 10, 2026)
# Sid realized that refreshing pages store the current
# state of the attempt (i.e. continues where the student left off) but with a new visitId
# There is also the case where simultaneous tabs are open, where,
#   attemptHlc point to the same attempt but different visitIds.
# In other words,
#       2+ tabs => same attemptHlc but different visitIds and different problem-states.
#       1 tab, but student refreshes, or comes back to it later =>
#           same attemptHlc, different visitIds but same problem-state (i.e. continuation).


def _collapse_visit_ids_if_no_backtracking(visit_ids: pd.Series) -> pd.Series:
    """Collapse visit ids to the first one if the sequence never returns to an earlier id.

    Examples:
        a,a,a,b,b -> a,a,a,a,a
        a,a,b,a   -> unchanged
    """
    if visit_ids.empty:
        return visit_ids

    first_visit_id = visit_ids.iloc[0]
    if pd.isna(first_visit_id):
        return visit_ids

    seen_visit_ids = {first_visit_id}
    previous_visit_id = first_visit_id
    has_changed = False

    for current_visit_id in visit_ids.iloc[1:]:
        if current_visit_id == previous_visit_id:
            continue
        has_changed = True
        if current_visit_id in seen_visit_ids:
            # Back-and-forth detected, keep original sequence.
            return visit_ids
        seen_visit_ids.add(current_visit_id)
        previous_visit_id = current_visit_id

    if not has_changed:
        return visit_ids

    collapsed_visit_ids = visit_ids.copy()
    collapsed_visit_ids.loc[:] = first_visit_id
    return collapsed_visit_ids


def _normalize_visit_ids(problem_event_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize visit ids per attempt when visit sequence is one-way only."""
    group_cols = ["studentSessionId", "taskNumber", "attemptHlc"]
    normalized_df = problem_event_df.copy()
    normalized_df["visitId"] = normalized_df.groupby(
        group_cols,
        observed=True,
        sort=False,
    )["visitId"].transform(_collapse_visit_ids_if_no_backtracking)
    return normalized_df


def _handle_single_problem_logs(
    args: tuple[str, str, str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate event log data to attempt level and student-problem level for a single problem.

    Args:
        args (tuple[str, str, str, pd.DataFrame]): Tuple containing
            task number, startState, goalState,
            and DataFrame with event log data for that task.


    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames aggregated to attempt level and student-problem level for a single problem.

    """
    task_number, start_state, goal_state, problem_event_df = args

    problem_event_df = problem_event_df.sort_values(
        by=["studentSessionId", "taskNumber", "timestamp"]
    )

    problem_event_df = _normalize_visit_ids(problem_event_df)

    # get graph and student paths for problem
    _, student_G_dict, student_paths = make_problem_graph(
        problem_event_df, task_number, start_state, goal_state, store_attempt_meta=True
    )

    # get classification for attempt
    class_df = process_problem_classification(
        student_paths, student_G_dict, start_state, goal_state
    )
    # merge with next
    # aggregations for attempt and student-problem levels
    attempt_agg_df, student_problem_df = _aggregate_single_problem_logs(
        problem_event_df
    )
    attempt_agg_df = attempt_agg_df.merge(
        class_df,
        on=["studentSessionId", "attemptHlc", "visitId"],
        how="left",
    )

    return attempt_agg_df, student_problem_df


def _aggregate_single_problem_logs(
    problem_event_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate event log data for a single problem.
        This also generates attempt classifications.

    Args:
        problem_event_df (pd.DataFrame): DataFrame containing event log data for a single problem.
        class_df (pd.DataFrame): DataFrame containing classification data for a single problem.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrame aggregated to attempt level and student-problem level for a single problem.

    """

    # remove redundant row
    problem_event_df: pd.DataFrame = problem_event_df.loc[
        ~problem_event_df["eventType"].isin(["visitGameScreen"])
    ].copy()
    # sort by studentRef, studentSessionId, taskNumber, timestamp
    problem_event_df = problem_event_df.sort_values(
        by=["studentSessionId", "taskNumber", "timestamp"]
    )

    # fix reset task belonging to the previous attempt
    # doing this will ensure time after completion or last action is not counted towards total_time
    problem_event_df[["next_attemptHlc", "next_reset_reason"]] = (
        problem_event_df.groupby(
            ["studentSessionId", "taskNumber"], observed=True, sort=False
        )[["attemptHlc", "reason"]].shift(-1)
    )
    # increment attemptHlc to next_attemptHlc for resetTask events
    # NOTE: Sid did this to allow the calculation of pause_time while grouping by attemptHlc
    #       there is no other reason to do this and it does not affect any other calculations
    reset_mask = problem_event_df["eventType"] == EventLogTypes.RESET_TASK
    problem_event_df.loc[reset_mask, "attemptHlc"] = problem_event_df.loc[
        reset_mask, "next_attemptHlc"
    ]

    # remove empty visits
    # mask for visits that contains at least one user interaction event
    no_empty_visits = problem_event_df.groupby(
        ["studentSessionId", "taskNumber", "attemptHlc", "visitId"],
        observed=True,
        sort=False,
    )["eventType"].transform(lambda s: s.isin(ACTION_EVENTS).any())
    no_empty_visits = no_empty_visits.astype(bool).fillna(True)
    problem_event_df = problem_event_df[no_empty_visits].reset_index(drop=True)

    # shift event timestamp and type by 1 to get previous event information for pause time calculation and attempt aggregation
    problem_event_df[["prev_timestamp", "prev_eventType"]] = problem_event_df.groupby(
        ["studentSessionId", "taskNumber", "attemptHlc", "visitId"],
        observed=True,
        sort=False,
    )[["timestamp", "eventType"]].shift(1)

    # compute delta time (time difference between current and previous event)
    problem_event_df["delta_time"] = (
        problem_event_df["timestamp"] - problem_event_df["prev_timestamp"]
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
        ["studentSessionId", "taskNumber", "attemptHlc", "visitId"],
        observed=True,
        sort=False,
    )

    # attempt aggregation
    attempt_agg_df = student_attempt_group.agg(
        studentRef=("studentRef", "first"),
        start_time=("timestamp", "first"),
        # end time needs to ignore time after solvingTask (e.g. reset)
        end_time=("timestamp", "last"),
        total_time=("delta_time", _sum_delta_times_under_5_min),
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
        num_keypad_errors=("mistake", lambda x: (x == EventMistakeTypes.KEYPAD).sum()),
        num_shaking_errors=("mistake", lambda x: (x == EventMistakeTypes.TAP).sum()),
        num_snapping_errors=("mistake", lambda x: (x == EventMistakeTypes.DRAG).sum()),
        stars=("stars", "max"),
        clicked_reset_button=(
            "next_reset_reason",
            lambda x: (x == ResetReasons.RESET_BUTTON).any(),
        ),
        clicked_retry_button=(
            "next_reset_reason",
            lambda x: (x == ResetReasons.RETRY_BUTTON).any(),
        ),
    ).reset_index()

    attempt_agg_df.sort_values(
        ["studentSessionId", "taskNumber", "attemptHlc", "start_time"],
        inplace=True,
    )
    # add corrected attempt number
    attempt_agg_df["attempt_number"] = attempt_agg_df.groupby(
        ["studentSessionId", "taskNumber"], observed=True, sort=False
    )["attemptHlc"].transform(lambda s: (~s.duplicated()).cumsum())
    # add attempt visit number (visit number within the same attempt)
    # this occurs when there are multiple visits for the same attempt (multi-tab)
    attempt_agg_df["attempt_visit_number"] = (
        attempt_agg_df.groupby(
            [
                "studentSessionId",
                "taskNumber",
                "attemptHlc",
                "attempt_number",
            ],
            observed=True,
            sort=False,
        ).cumcount()
        + 1
    )

    attempt_agg_df["pause_time_ratio"] = (
        attempt_agg_df["pause_time"] / attempt_agg_df["total_time"]
    )
    attempt_agg_df["replay_attempt"] = attempt_agg_df.groupby(
        ["studentSessionId", "taskNumber"], observed=True, sort=False
    )["completed_dur_attempt"].transform(_get_replay_attempt)

    # set dtypes
    attempt_agg_df = _set_attempt_dtypes(attempt_agg_df)
    # aggregate to student-problem level
    student_problem_df = _aggregate_single_student_problem(attempt_agg_df)
    return attempt_agg_df, student_problem_df


def _set_attempt_dtypes(attempt_agg_df: pd.DataFrame) -> pd.DataFrame:
    return attempt_agg_df.astype(ATTEMPT_TYPE_MAPPING)


def _get_pause_time(col: pd.Series) -> float | None:
    col = col.apply(pd.Series)  # expand tuples  # ty:ignore[invalid-assignment]
    action_times: pd.Series[pd.Timedelta] = col.loc[col[1] == True, 0]

    if len(action_times) == 0:  # no actions for this attempt
        return None
    else:
        pause_time: float = action_times.iloc[0].total_seconds()
        if (
            pause_time >= 5 * 60
        ):  # if pause time is greater than-eq to 5 minutes, return None
            return None
        else:
            return pause_time


def _sum_delta_times_under_5_min(delta_times: pd.Series) -> float | None:
    """Sum delta times that are less than 5 minutes."""
    # Filter delta times less than 5 minutes (300 seconds)
    valid_times: pd.Series[pd.Timedelta] = delta_times[
        delta_times < pd.Timedelta(minutes=5)
    ]

    if len(valid_times) == 0:
        return None
    else:
        return valid_times.sum().total_seconds()


def _get_validity_first_step(col: pd.Series) -> bool | None:
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
