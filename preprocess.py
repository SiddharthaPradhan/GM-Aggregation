"""
Preprocessing script.
Most of these functions are simple column removals and filtering.
Functions can be extended to add more complex preprocessing if needed.
"""

import pandas as pd
import json
from zipfile import ZipExtFile, ZipFile
import logging
from utils import (
    write_to_study_meta_text,
    save_df,
    load_df_from_file,
    get_file,
    OUTPUT_TYPE,
)

logger = logging.getLogger(__name__)


# TODO:
# - Fix dtype in dataframes before saving


def preprocess_and_save_event_log(
    input: str | ZipFile, output_dir: str, output_type: OUTPUT_TYPE
) -> pd.DataFrame:
    """Preprocess event log data from GMA, i.e. event-logs.json from the research-data endpoint.
    Return preprocessed event log dataframe for aggregation."""
    logger.info("Starting Event Log Preprocessing..")
    event_log_df = load_df_from_file(get_file(input, "event_logs"))
    columns_to_remove = ["_id", "uid"]
    event_log_df = event_log_df.drop(columns=columns_to_remove, errors="raise")
    save_df(event_log_df, "event_logs", output_dir, output_type=output_type)
    return event_log_df


def preprocess_gma_attempt_data(gma_attempt_df: pd.DataFrame):
    """Preprocess attempt data from GMA, i.e. attempt-data.json from the research-data endpoint."""
    columns_to_remove = ["_id", "classlessCanvas"]
    gma_attempt_df = gma_attempt_df.drop(columns=columns_to_remove, errors="raise")
    return gma_attempt_df


def preprocess_task_metadata(task_meta_df: pd.DataFrame):
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
    input: str | ZipFile, output_dir: str, output_type: OUTPUT_TYPE
):
    """Preprocess and save metadata, i.e., Roster, Task, and Study
    Study metadata is saved as a simple .txt file for quick reference.
    Args:
        input (str | ZipExtFile): Location of input data, either a directory or a ZipFile object
        output_dir (str): Directory to save the preprocessed metadata
        output_type (OUTPUT_TYPE): Format to save the preprocessed metadata
    """
    # Roster Metadata
    logger.info("Starting Metadata Preprocessing..")
    try:
        roster_meta_df = load_df_from_file(get_file(input, "roster"))
        roster_meta_df = preprocess_roster_metadata(roster_meta_df)
        logger.info(f"Number of Students in Roster: {roster_meta_df.shape[0]}")
        logger.info("Saving Roster Metadata...")
        logger.debug(f"Roster Metadata Columns: {roster_meta_df.columns}")

        if roster_meta_df is not None:  # save only non-public-session studies
            save_df(
                roster_meta_df,
                "roster_metadata",
                output_dir,
                output_type=output_type,
            )
        task_meta_df = load_df_from_file(get_file(input, "task"))
        task_meta_df = preprocess_task_metadata(task_meta_df)
        logger.info(f"Number of Tasks: {task_meta_df.shape[0]}")
        logger.info("Saving Task Metadata...")
        logger.debug(f"Task Metadata Columns: {task_meta_df.columns}")
        save_df(
            task_meta_df,
            "task_metadata",
            output_dir,
            output_type=output_type,
        )
    except Exception as e:
        logger.error(f"Error while preprocessing metadata: {e}")
