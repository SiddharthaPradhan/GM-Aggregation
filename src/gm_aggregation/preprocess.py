"""
Preprocessing script.
Most of these functions are simple column removals and filtering.
Functions can be extended to add more complex preprocessing if needed.
"""

import pandas as pd
import json
import re
from zipfile import ZipExtFile, ZipFile
import logging
import multiprocessing as mp
from typing import Callable, TypedDict
from tqdm import tqdm
from .utils import (
    write_to_study_meta_text,
    save_df,
    load_df_from_file,
    get_file,
    OUTPUT_TYPE,
    EventLogTypes,
)
from py_asciimath.translator.translator import Tex2ASCIIMath, ASCIIMath2Tex

tex2ascii_math = Tex2ASCIIMath(log=False, inplace=False)
ascii2tex_math = ASCIIMath2Tex(log=False, inplace=False)
logging.getLogger().setLevel(logging.ERROR)  # suppress py_asciimath logging
logger = logging.getLogger("GM-Aggregator." + __name__)


class ProgressEvent(TypedDict, total=False):
    stage: str
    completed: int
    total: int
    message: str


ProgressCallback = Callable[[ProgressEvent], None]

VALID_ACTIONS = [
    EventLogTypes.MATH_STEP,
    EventLogTypes.UNDO_STEP,
    EventLogTypes.REDO_STEP,
    EventLogTypes.MATH_MISTAKE,
]

EVENT_LOG_TYPE_MAPPING = {
    "studentRef": "category",
    "sessionId": "category",
    "taskNumber": "uint16[pyarrow]",
    "attemptNumber": "uint16[pyarrow]",
    "visitId": "category",
    "timestamp": "datetime64[ms]",
    "studentSessionId": "category",
    "eventType": "category",
    "attemptHlc": "category",
    "actionId": "category",
    "steps": "uint16[pyarrow]",
    "stars": "uint8[pyarrow]",
    "mistake": "category",
    "wasSolved": "bool[pyarrow]",
    "reason": "category",
}


def convert_state(state, to_latex, converter):
    if pd.isna(state):
        return state
    else:
        try:
            # try to translate between LaTeX and ASCIIMath
            translated_state = converter.translate(state, from_file=False)
            # remove $..$ as ASCIIMath2Tex wraps the whole expression in $..$
            # also strip specific whitespaces to make it consistent with GMath logging
            if to_latex:
                # the following is needed to match the formatting of the latex converted strings
                # to the ones logged by GMath, which have their own internal converter/fomatting
                # it is computationally cheaper to do string replacements
                # than to use ASTs and compare the structure
                translated_state = translated_state.lstrip("$").rstrip("$")
                translated_state = translated_state.replace(" ", "")
                # Match GMath formatting where numeric coefficients are adjacent to variables,
                # e.g. 15\cdot c+10 -> 15c+10 and 3\cdot 5\cdot c -> 3\cdot 5c.
                translated_state = re.sub(
                    r"(\d+)\\cdot(?=[A-Za-z])",
                    r"\1",
                    translated_state,
                )
                translated_state = re.sub(
                    r"(?<=[A-Za-z])\\cdot(?=[A-Za-z])",
                    "",
                    translated_state,
                )
                translated_state = translated_state.replace("\\cdot", "\\cdot ")
            else:
                # remove whitespace in the asciimath
                translated_state = translated_state.replace(" ", "")
            return translated_state
        except Exception as e:
            logger.warning(
                f"Error translating LaTeX to ASCIIMath for state: {state}. Error: {e}"
            )
            return state


def handle_latex_in_partial_event_log(
    partial_event_log_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert LaTeX in oldState and newState columns to ASCIIMath for better
    readability and easier parsing later on."""
    partial_event_log_df["oldState_latex"] = partial_event_log_df["oldState"].copy()
    partial_event_log_df["newState_latex"] = partial_event_log_df["newState"].copy()

    partial_event_log_df.loc[
        partial_event_log_df["eventType"].isin(VALID_ACTIONS), "oldState"
    ] = partial_event_log_df.loc[
        partial_event_log_df["eventType"].isin(VALID_ACTIONS), "oldState"
    ].transform(
        lambda s: convert_state(s, to_latex=False, converter=tex2ascii_math)
    )

    partial_event_log_df.loc[
        partial_event_log_df["eventType"].isin(VALID_ACTIONS), "newState"
    ] = partial_event_log_df.loc[
        partial_event_log_df["eventType"].isin(VALID_ACTIONS), "newState"
    ].transform(
        lambda s: convert_state(s, to_latex=False, converter=tex2ascii_math)
    )
    return partial_event_log_df


def handle_latex_in_event_log(
    event_log_df: pd.DataFrame,
    n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    # all task numbers
    task_numbers = event_log_df["taskNumber"].unique()
    task_numbers = task_numbers[~pd.isna(task_numbers)]
    total_tasks = len(task_numbers)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "event_log_preprocessing",
                "completed": 0,
                "total": total_tasks,
                "message": "Starting event log preprocessing",
            }
        )
    mp_jobs = []
    for task_number in task_numbers:
        mp_jobs.append(
            event_log_df.loc[event_log_df["taskNumber"] == task_number].copy(),
        )
        if n_jobs == -1:
            num_processes = None  # Use all available cores
        else:
            num_processes = min(n_jobs, len(mp_jobs))
        df_list_event_log = []
    with (
        mp.get_context("spawn").Pool(processes=num_processes) as pool,
        tqdm(
            total=len(mp_jobs),
            desc="Preprocessing States in Event Log",
            unit="problem",
            disable=None,
        ) as pbar,
    ):
        for completed, result in enumerate(
            pool.imap_unordered(handle_latex_in_partial_event_log, mp_jobs), start=1
        ):
            df_list_event_log.append(result)
            pbar.update()
            pbar.refresh()
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "event_log_preprocessing",
                        "completed": completed,
                        "total": total_tasks,
                    }
                )
    event_log_df: pd.DataFrame = pd.concat(df_list_event_log, ignore_index=True)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "event_log_preprocessing",
                "completed": total_tasks,
                "total": total_tasks,
                "message": "Finished event log preprocessing",
            }
        )
    return event_log_df


def preprocess_event_log(
    event_log_df: pd.DataFrame,
    convert_latex: bool = False,
    n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    columns_to_remove = ["_id", "uid"]
    event_log_df = event_log_df.drop(columns=columns_to_remove, errors="raise")
    event_log_df["timestamp"] = pd.to_datetime(event_log_df["timestamp"])
    # For consistency with previous MFL versions, we will convert Latex states into ASCIIMath.
    # the latex column is retained.
    if convert_latex:
        event_log_df = handle_latex_in_event_log(
            event_log_df, n_jobs=n_jobs, progress_callback=progress_callback
        )
    elif progress_callback is not None:
        progress_callback(
            {
                "stage": "event_log_preprocessing",
                "completed": 1,
                "total": 1,
                "message": "Skipping LaTeX conversion",
            }
        )

    event_log_df.sort_values(
        by=["studentSessionId", "taskNumber", "timestamp"],
        inplace=True,
        ignore_index=True,
    )
    event_log_df = set_event_log_types(event_log_df)
    return event_log_df


def preprocess_and_save_event_log(
    input: str | ZipFile,
    output_dir: str,
    output_type: OUTPUT_TYPE,
    convert_latex: bool = False,
    n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Preprocess event log data from GMA, i.e. event-logs.json from the research-data endpoint.
    Return preprocessed event log dataframe for aggregation."""
    logger.info("Starting Event Log Preprocessing..")
    event_log_df = load_df_from_file(get_file(input, "event_logs"))
    event_log_df = preprocess_event_log(
        event_log_df,
        convert_latex=convert_latex,
        n_jobs=n_jobs,
        progress_callback=progress_callback,
    )

    save_df(event_log_df, "event_logs", output_dir, output_type=output_type)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "event_log_preprocessing",
                "completed": 1,
                "total": 1,
                "message": "Finished event log preprocessing",
            }
        )
    return event_log_df


def set_event_log_types(event_log_df: pd.DataFrame) -> pd.DataFrame:
    """Set appropriate data types for event log dataframe."""
    event_log_df = event_log_df.astype(EVENT_LOG_TYPE_MAPPING)
    return event_log_df


def preprocess_gma_attempt_data(gma_attempt_df: pd.DataFrame):
    """Preprocess attempt data from GMA, i.e. attempt-data.json from the research-data endpoint."""
    columns_to_remove = ["_id", "classlessCanvas"]
    gma_attempt_df = gma_attempt_df.drop(columns=columns_to_remove, errors="raise")
    return gma_attempt_df


def preprocess_task_metadata(
    task_meta_df: pd.DataFrame, convert_latex: bool
) -> pd.DataFrame:
    """Preprocess task metadata and keep only information from the latest session.

    Args:
        task_meta_df (pd.DataFrame): Raw task metadata dataframe.
    """
    # find the sessionId relating to the latest added session.
    # this should have the most up-to-date task metadata.
    latest_session_id = task_meta_df.loc[
        task_meta_df["sessionCreatedAt"] == task_meta_df["sessionCreatedAt"].max(),
        "sessionId",
    ].iloc[0]
    task_meta_df = task_meta_df.loc[task_meta_df["sessionId"] == latest_session_id]
    columns_to_remove = [
        "sessionCode",
        "taskId",
        "chapterId",
        "sessionId",
        "sessionCreatedAt",
        "hintDelta",
    ]
    task_meta_df = task_meta_df.drop(columns=columns_to_remove, errors="raise")
    # task_meta_df has states in asciimath (IDK ask GM team why)
    if not convert_latex:
        # convert so we can string compare
        task_meta_df["startState_asciimath"] = task_meta_df["startState"].copy()
        task_meta_df["goalState_asciimath"] = task_meta_df["goalState"].copy()
        task_meta_df["startState"] = task_meta_df["startState"].transform(
            lambda s: convert_state(s, to_latex=True, converter=ascii2tex_math)
        )
        task_meta_df["goalState"] = task_meta_df["goalState"].transform(
            lambda s: convert_state(s, to_latex=True, converter=ascii2tex_math)
        )

    return task_meta_df


def preprocess_roster_metadata(roster_meta_df: pd.DataFrame) -> pd.DataFrame | None:
    """Preprocess roster metadata

    Args:
        roster_meta_df (pd.DataFrame): Raw roster metadata dataframe.
    """
    # Roster Metadata can be empty if the data is from a public session.
    # If so print a warning and return None
    if roster_meta_df.empty:
        logger.warning(
            "Roster metadata is empty. This data may be from a public session."
        )
        return None
    return roster_meta_df


def save_study_metadata(input: str | ZipFile, output_dir: str):
    """Save study metadata as a simple .txt file for quick reference.
    Additional study-level information is saved during aggregation.

    """
    study_file = get_file(input, "study")
    study_metadata = json.load(study_file)

    # Format as plain text without JSON structure or quotes
    text_content = ""
    for key, value in study_metadata.items():
        text_content += f"{key}: {value}\n"

    write_to_study_meta_text(output_dir, text_content, append=False)


def preprocess_and_save_metadata(
    input: str | ZipFile, output_dir: str, output_type: OUTPUT_TYPE, convert_latex: bool
) -> pd.DataFrame | None:
    """Preprocess and save metadata, i.e., Roster, Task, and Study
    Study metadata is saved as a simple .txt file for quick reference.
    Args:
        input (str | ZipExtFile): Location of input data, either a directory or a ZipFile object
        output_dir (str): Directory to save the preprocessed metadata
        output_type (OUTPUT_TYPE): Format to save the preprocessed metadata
        convert_latex (bool): Whether to convert LaTeX to ASCII math
    """
    # save initial study metadata
    # other study-level info will be added during aggregation
    save_study_metadata(input, output_dir)
    # Roster Metadata
    logger.info("Starting Metadata Preprocessing..")
    try:
        roster_meta_df = load_df_from_file(get_file(input, "roster"))
        roster_meta_df = preprocess_roster_metadata(roster_meta_df)

        if roster_meta_df is not None:  # save only non-public-session studies
            logger.info(f"Number of Students in Roster: {roster_meta_df.shape[0]}")
            logger.info("Saving Roster Metadata...")
            logger.debug(f"Roster Metadata Columns: {roster_meta_df.columns}")
            save_df(
                roster_meta_df,
                "roster_metadata",
                output_dir,
                output_type=output_type,
            )
        task_meta_df = load_df_from_file(get_file(input, "task"))
        task_meta_df = preprocess_task_metadata(task_meta_df, convert_latex)
        logger.info(f"Number of Tasks: {task_meta_df.shape[0]}")
        logger.info("Saving Task Metadata...")
        logger.debug(f"Task Metadata Columns: {task_meta_df.columns}")
        save_df(
            task_meta_df,
            "task_metadata",
            output_dir,
            output_type=output_type,
        )
        return task_meta_df
    except Exception as e:
        logger.error(f"Error while preprocessing metadata: {e}")
        raise e
