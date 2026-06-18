"""
Preprocessing script.
Most of these functions are simple column removals and filtering.
Functions can be extended to add more complex preprocessing if needed.
"""

import os
import pandas as pd
import json
import re
import gc
import tempfile
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
    CONTENT_DICT,
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
    dask_client=None,
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
    df_list_event_log = []
    tmp_dir = None

    if dask_client is not None:
        from dask.distributed import as_completed as dask_as_completed

        tmp_dir = tempfile.mkdtemp(prefix="latex-event-log-")
        file_counter = [0]

        n_concurrent = len(dask_client.nthreads()) * 2
        task_iter = iter(task_numbers)

        def _submit_next():
            try:
                tn = next(task_iter)
                return dask_client.submit(
                    handle_latex_in_partial_event_log,
                    event_log_df.loc[event_log_df["taskNumber"] == tn].copy(),
                    pure=False,
                )
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
            desc="Preprocessing States in Event Log",
            unit="problem",
            disable=None,
        ) as pbar:
            for completed, future in enumerate(ac, start=1):
                result_df = future.result()
                file_path = os.path.join(tmp_dir, f"event_log_{file_counter[0]}.csv")
                result_df.to_csv(file_path, index=False)
                file_counter[0] += 1
                del result_df
                gc.collect()
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
                next_f = _submit_next()
                if next_f is not None:
                    ac.add(next_f)
    else:
        mp_jobs = []
        for task_number in task_numbers:
            mp_jobs.append(
                event_log_df.loc[event_log_df["taskNumber"] == task_number].copy(),
            )
        num_processes = None if n_jobs == -1 else min(n_jobs, len(mp_jobs))
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

    try:
        if tmp_dir is not None:
            file_paths = sorted(
                [
                    os.path.join(tmp_dir, f)
                    for f in os.listdir(tmp_dir)
                    if f.endswith(".csv")
                ]
            )

            all_chunks = []
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                all_chunks.append(df)
                if len(all_chunks) >= 4:
                    intermediate_df = pd.concat(all_chunks, ignore_index=True)
                    del all_chunks
                    all_chunks = [intermediate_df]
                    gc.collect()

            event_log_df = (
                pd.concat(all_chunks, ignore_index=True)
                if all_chunks
                else pd.DataFrame()
            )
            gc.collect()
        else:
            event_log_df = pd.concat(df_list_event_log, ignore_index=True)

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
    finally:
        if tmp_dir is not None:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


def _discover_ndjson_columns(path: str) -> list[str]:
    """Scan an NDJSON file once to collect the union of all keys (in first-seen
    order) across every record, without materializing any rows.

    GMA event log records serialise their keys in different orders depending
    on event type, which breaks dd.read_json's schema inference (it infers a
    fixed column order from the first chunk and rejects any later chunk whose
    inferred order differs, even though the same columns are all present).
    Scanning once up front with json.loads is cheap (a few seconds even for
    1M+ lines) and gives a single canonical column order every chunk can be
    reindexed to.
    """
    columns: dict[str, None] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in record.keys():
                columns.setdefault(key, None)
    return list(columns.keys())


def _ndjson_chunk_boundaries(path: str, blocksize: int) -> list[tuple[int, int]]:
    """Compute (start, end) byte offsets covering *path*, each aligned to a
    complete NDJSON line (never splitting a JSON record across chunks)."""
    file_size = os.path.getsize(path)
    boundaries = []
    with open(path, "rb") as f:
        pos = 0
        while pos < file_size:
            end = min(pos + blocksize, file_size)
            if end < file_size:
                f.seek(end)
                f.readline()  # consume the partial line to land on a boundary
                end = f.tell()
            boundaries.append((pos, end))
            pos = end
    return boundaries


def _read_ndjson_block(
    path: str, start: int, end: int, columns: list[str]
) -> pd.DataFrame:
    """Read one byte-range chunk of an NDJSON file and reindex it to *columns*
    so every chunk has an identical, predictable column order regardless of
    the order keys happened to be serialised in for that chunk's records."""
    from io import BytesIO

    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    df = pd.read_json(BytesIO(data), lines=True)

    return df.reindex(columns=columns)


_NUMERIC_RAW_COLUMNS = {"taskNumber", "attemptNumber", "steps", "stars", "dx", "dy"}
_BOOLEAN_RAW_COLUMNS = {"wasSolved", "dropClosestToSelf"}

# Columns genuinely required downstream (generate_graph.py reads
# row.oldState/row.newState/row.actionName per event) but absent from
# EVENT_LOG_TYPE_MAPPING, which only lists columns kept through to the
# final typed output. _fill_optional_columns must protect these too, or a
# record set that's missing one of them entirely raises a KeyError /
# AttributeError deep in graph building rather than failing gracefully.
_REQUIRED_UNTYPED_COLUMNS = {
    "oldState": "object",
    "newState": "object",
    "actionName": "object",
}


def _raw_csv_dtype_map(columns) -> dict:
    """Explicit dtype overrides for the preprocessed event log.

    Plain numpy/pandas dtypes (not pyarrow ones) avoid a null-type mismatch
    seen earlier: a chunk where an optional column like 'mistake' is
    entirely empty infers as null[pyarrow], while a chunk with real values
    infers as string[pyarrow] — float64 naturally represents missing
    numeric values as NaN instead, and "boolean" is pandas' own nullable
    boolean extension type.

    Columns marked "category" in EVENT_LOG_TYPE_MAPPING use that here too:
    low-cardinality, highly-repeated string columns (eventType, attemptHlc,
    studentSessionId, mistake, reason, etc.) are stored once per unique
    value plus a small integer code per row, instead of a full Python
    string object per row — the single biggest lever for this data's
    memory footprint, since most of these ~28 columns are otherwise object
    dtype. It's safe to apply per chunk here (each chunk gets its own local
    category set) because, unlike the earlier Dask DataFrame design, there
    is no cross-chunk meta validation requiring identical categories.
    """
    dtype_map = {}
    for col in columns:
        if col == "timestamp":
            continue  # parsed separately via dd.to_datetime
        elif col in _NUMERIC_RAW_COLUMNS:
            dtype_map[col] = "float64"
        elif col in _BOOLEAN_RAW_COLUMNS:
            dtype_map[col] = "boolean"
        elif EVENT_LOG_TYPE_MAPPING.get(col) == "category":
            dtype_map[col] = "category"
        else:
            dtype_map[col] = "object"
    return dtype_map


class TaskSplitEventLog:
    """Marker object returned by preprocess_and_save_event_log in place of a
    DataFrame: maps each taskNumber to the path of a small CSV file
    containing only that task's rows, plus the directory holding them (for
    cleanup once aggregation is done reading them).

    This avoids two failure modes seen with a single Dask-backed lazy
    DataFrame: persist() must hold every partition in the worker's memory
    at once (OOM risk on a tight budget), while staying fully lazy means
    re-scanning the *entire* event log from disk once per task during
    aggregation (catastrophically slow — O(n_tasks * file_size) instead of
    O(file_size)). Splitting once up front is a single linear pass, and
    each task then reads only its own small file.
    """

    def __init__(self, task_files: dict, split_dir: str, dtype_map: dict):
        self.task_files = task_files
        self.split_dir = split_dir
        self.dtype_map = dtype_map

    def read_task(self, task_number) -> pd.DataFrame:
        df = pd.read_csv(self.task_files[task_number], dtype=self.dtype_map)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        return df

    def cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.split_dir, ignore_errors=True)


def _process_event_log_single_pass(
    ndjson_path: str,
    output_dir: str,
    output_type: OUTPUT_TYPE,
    blocksize: int = 32 * 1024 * 1024,
) -> "TaskSplitEventLog":
    """Read the raw NDJSON event log exactly once, chunk by chunk, and in the
    same pass: preprocess each chunk, write it to the canonical CSV (if
    requested), write it to SQLite (if requested), and split it into
    per-task files for aggregation to read from later.
    """
    import sqlite3
    import tempfile

    columns = _discover_ndjson_columns(ndjson_path)
    boundaries = _ndjson_chunk_boundaries(ndjson_path, blocksize)

    write_csv = output_type in ("csv", "both")
    write_sqlite = output_type in ("sqlite", "both")
    working_csv_path = os.path.join(output_dir, "event_logs.csv")

    split_dir = tempfile.mkdtemp(prefix="task-split-")
    open_task_files: dict = {}  # task_number -> (file_path, file_handle)
    dtype_map: dict | None = None
    csv_file = None
    sqlite_conn = None
    csv_header_written = False
    sqlite_first_write = True

    try:
        if write_csv:
            csv_file = open(working_csv_path, "w", newline="")
        if write_sqlite:
            sqlite_conn = sqlite3.connect(os.path.join(output_dir, "GMA_data.db"))

        for start, end in boundaries:
            chunk = _read_ndjson_block(ndjson_path, start, end, columns)
            chunk = chunk.drop(columns=["_id", "uid"], errors="ignore")
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="mixed")
            chunk = _fill_optional_columns(chunk)

            if dtype_map is None:
                dtype_map = _raw_csv_dtype_map(list(chunk.columns))

            for col, dtype in dtype_map.items():
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(dtype)

            if csv_file is not None:
                chunk.to_csv(csv_file, index=False, header=not csv_header_written)
                csv_header_written = True

            if sqlite_conn is not None:
                chunk.to_sql(
                    "event_logs",
                    sqlite_conn,
                    if_exists="replace" if sqlite_first_write else "append",
                    index=False,
                )
                sqlite_first_write = False

                sqlite_conn.commit()

            for task_number, group in chunk.groupby("taskNumber"):
                if task_number not in open_task_files:
                    file_path = os.path.join(split_dir, f"task_{task_number}.csv")
                    f = open(file_path, "w", newline="")
                    group.to_csv(f, index=False)
                    open_task_files[task_number] = (file_path, f)
                else:
                    _, f = open_task_files[task_number]
                    group.to_csv(f, index=False, header=False)
    finally:
        if csv_file is not None:
            csv_file.close()
        if sqlite_conn is not None:
            sqlite_conn.close()
        for _, f in open_task_files.values():
            f.close()

    task_files = {tn: fp for tn, (fp, _) in open_task_files.items()}
    return TaskSplitEventLog(task_files, split_dir, dtype_map or {})


def _fill_optional_columns(df):
    """Add any columns from EVENT_LOG_TYPE_MAPPING that are absent in df as all-NA.

    Works on both pandas DataFrames and Dask DataFrames. Pandas columns are
    initialised with the correct nullable pyarrow dtype; Dask columns are set to
    a scalar None (the dtype is resolved when each partition is computed).
    """
    try:
        import dask.dataframe as _dd

        is_dask = isinstance(df, _dd.DataFrame)
    except ImportError:
        is_dask = False

    for col, dtype in {**EVENT_LOG_TYPE_MAPPING, **_REQUIRED_UNTYPED_COLUMNS}.items():
        if col not in df.columns:
            if is_dask:
                df[col] = None
            else:
                df[col] = pd.array([pd.NA] * len(df), dtype=dtype)
    return df


def preprocess_event_log(
    event_log_df: pd.DataFrame,
    convert_latex: bool = False,
    n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
    dask_client=None,
) -> pd.DataFrame:
    columns_to_remove = ["_id", "uid"]
    event_log_df = event_log_df.drop(columns=columns_to_remove, errors="raise")
    event_log_df["timestamp"] = pd.to_datetime(event_log_df["timestamp"])
    # For consistency with previous MFL versions, we will convert Latex states into ASCIIMath.
    # the latex column is retained.
    if convert_latex:
        event_log_df = handle_latex_in_event_log(
            event_log_df,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
            dask_client=dask_client,
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

    event_log_df = _fill_optional_columns(event_log_df)
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
    dask_client=None,
):
    """Preprocess event log data from GMA, i.e. event-logs.json from the research-data endpoint.
    Returns a TaskSplitEventLog (per-task files already split out on disk) when a Dask client
    is provided, this is what aggregate_event_log reads from. For zip inputs the event log is
    streamed to a temp file first since processing needs a file path. Falls back to a pandas
    DataFrame only when LaTeX conversion is needed.
    """
    logger.info("Starting Event Log Preprocessing..")

    use_single_pass = dask_client is not None and not convert_latex

    if use_single_pass:
        import shutil
        import tempfile

        tmp_path = None
        if isinstance(input, ZipFile):
            # The single-pass processor needs a file path, not a file object.
            # Stream the zip member to a temp file so we never load it all
            # into memory at once (shutil.copyfileobj copies in 16KB chunks).
            tmp = tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False)
            try:
                with input.open(CONTENT_DICT["event_logs"]) as zf:
                    shutil.copyfileobj(zf, tmp)
            finally:
                tmp.close()
            tmp_path = tmp.name
            event_log_path = tmp_path
        else:
            event_log_path = os.path.join(input, CONTENT_DICT["event_logs"])

        try:
            task_split = _process_event_log_single_pass(
                event_log_path, output_dir, output_type
            )
        finally:
            if tmp_path is not None:
                os.unlink(tmp_path)

        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "event_log_preprocessing",
                    "completed": 1,
                    "total": 1,
                    "message": "Finished event log preprocessing",
                }
            )
        return task_split

    # Fallback: LaTeX conversion: use pandas
    event_log_df = load_df_from_file(get_file(input, "event_logs"), lines=True)
    event_log_df = preprocess_event_log(
        event_log_df,
        convert_latex=convert_latex,
        n_jobs=n_jobs,
        progress_callback=progress_callback,
        dask_client=dask_client,
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
        roster_meta_df = load_df_from_file(get_file(input, "roster"), lines=False)
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
        task_meta_df = load_df_from_file(get_file(input, "task"), lines=True)
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
