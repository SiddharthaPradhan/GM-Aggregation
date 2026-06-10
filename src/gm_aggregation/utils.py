import os
import pandas as pd
import zipfile
import sqlite3
from typing import Literal, get_args, IO
import json
from io import TextIOBase

OUTPUT_TYPE = Literal["sqlite", "csv", "both"]
FILE_KEY_TYPE = Literal["roster", "study", "task", "event_logs", "attempt_data"]
# Mapping of file keys to their respective filenames in the GMA data export
CONTENT_DICT = {
    "roster": "roster-metadata.json",
    "study": "study-metadata.json",
    "task": "task-metadata.json",
    "event_logs": "event-logs.json",
    "attempt_data": "attempt-data.json",
}
# Filename for the processed study metadata text file
STUDY_METADATA_TXT = "study_info.txt"
TXT_SEPARATOR = "-" * 60 + "\n"


# Event types in the event log
class EventLogTypes:
    MATH_STEP = "mathStep"
    MATH_MISTAKE = "mathMistake"
    SOLVED_TASK = "solvedTask"
    VISIT_TASK = "visitTask"
    RESET_TASK = "resetTask"
    SHOW_HINT = "showHint"
    VISIT_GAME_SCREEN = "visitGameScreen"
    HIDE_HINT = "hideHint"
    UNDO_STEP = "undo"
    REDO_STEP = "redo"


class EventMistakeTypes:
    KEYPAD = "keypad"
    TAP = "tap"
    DRAG = "drag"


# User interaction events: math actions, mistake events or hint events
ACTION_EVENTS = [
    EventLogTypes.MATH_STEP,
    EventLogTypes.MATH_MISTAKE,
    EventLogTypes.HIDE_HINT,
    EventLogTypes.SHOW_HINT,
]


class ResetReasons:
    RESET_BUTTON = "resetButton"
    RETRY_BUTTON = "retryButton"
    AUTO = "autoReset"


def write_to_study_meta_text(
    directory: str | os.PathLike, content: str, append: bool = True
) -> None:
    """Write content to a text file.

    Args:
        directory (str | os.PathLike): Directory where the text file will be saved.
        content (str): Content to write to the file.
        append (bool): Whether to append to the file or overwrite it.
    """
    mode = "a" if append else "w"
    with open(os.path.join(directory, STUDY_METADATA_TXT), mode) as f:
        f.write(TXT_SEPARATOR)
        f.write(content)


def check_existence(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")


def load_zip(zip_path):
    """Validate and load zip object. Note that this does not extract or load files to memory."""
    check_existence(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a valid zip file.")

    zip = zipfile.ZipFile(zip_path, "r")
    # check if all required files are present
    if not all(file in zip.namelist() for file in CONTENT_DICT.values()):
        raise ValueError(
            "Zip file is missing required files. Please put in the zip file directly from GMA."
        )
    return zip


def load_df_from_file(
    file: TextIOBase | IO[bytes] | zipfile.ZipExtFile,
    lines: bool = False,
):
    """Load dataframe from file

    Args:
        file (TextIOBase | zipfile.ZipExtFile): A readonly file object from directory or Zip
        lines (bool, optional): Whether the file is in NDJSON format. Defaults to False.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_json(file, lines=lines, dtype_backend="pyarrow")
    return df


def save_df(
    df: pd.DataFrame,
    df_name: str,
    save_path: str | os.PathLike,
    db_name: str = "GMA_data.db",
    output_type: OUTPUT_TYPE = "csv",
) -> None:
    """Save DataFrame to CSV, SQLite, or both.

    Args:
        df (pd.DataFrame): DataFrame to save.
        df_name (str): Name of the CSV file or SQLite table.
    Raises:
        ValueError: If the output type is unsupported.
    """
    if output_type not in get_args(OUTPUT_TYPE):
        raise ValueError(
            f"Unsupported output type {output_type}. Use 'csv', 'sqlite', or 'both'."
        )
    if output_type == "csv" or output_type == "both":
        df.to_csv(os.path.join(save_path, f"{df_name}.csv"), index=False)
    if output_type == "sqlite" or output_type == "both":
        with sqlite3.connect(os.path.join(save_path, db_name)) as conn:
            df.to_sql(df_name, conn, if_exists="replace", index=False)


def get_study_id(
    study_meta_file: str | TextIOBase | IO[bytes] | zipfile.ZipExtFile,
) -> str:
    """Returns the study ID from the study metadata file in the input location or zip.

    Args:
            study_meta_file (str | TextIOBase | IO[bytes] | zipfile.ZipExtFile): Study metadata file path or readonly file object
    )
    """
    study_metadata = json.load(study_meta_file)
    return study_metadata["studyId"]


def get_file(
    input: str | zipfile.ZipFile, file_key: FILE_KEY_TYPE
) -> TextIOBase | IO[bytes] | zipfile.ZipExtFile:
    """Returns absolute file path or reader object for the `file_key` file from the input location or zip.

    Args:
        input (str | zipfile.ZipFile): Input directory or ZipFile
        file_key (str): Key for the desired file in the content dictionary

    Returns:
        TextIOBase | IO[bytes] | zipfile.ZipExtFile: Readonly file object
    """
    if isinstance(input, zipfile.ZipFile):
        return input.open(CONTENT_DICT[file_key])  # Readonly file object
    else:
        return open(os.path.join(input, CONTENT_DICT[file_key]), "r")  # File Path
