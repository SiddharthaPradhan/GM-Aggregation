import pandas as pd
from utils import load_zip, CONTENT_DICT, OUTPUT_TYPE


def aggregate_event_log(event_log_df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """Aggregate event log data to attempt level.

    Args:
        event_log_df (pd.DataFrame): DataFrame containing preprocessed event log data.
        n_jobs (int, optional): Number of parallel jobs to use. Defaults to -1 (use all available cores).
    Returns:
        pd.DataFrame: DataFrame aggregated to attempt level.
    """
    # Example aggregation logic (to be replaced with actual logic)
    attempt_level_df = (
        event_log_df.groupby("attemptId")
        .agg(
            total_events=pd.NamedAgg(column="eventType", aggfunc="count"),
            first_event_time=pd.NamedAgg(column="eventTime", aggfunc="min"),
            last_event_time=pd.NamedAgg(column="eventTime", aggfunc="max"),
        )
        .reset_index()
    )
    return attempt_level_df


def aggregate(attempt_level_df: pd.DataFrame, output_type: OUTPUT_TYPE = "sqlite"):
    """Aggregates the preprocessed data to the student-problem, student and problem levels.

    Args:
        attempt_level_df (pd.DataFrame): DataFrame containing preprocessed attempt-level data.
        output_type (OUTPUT_TYPE, optional): Whether to save as csv or in sqlite. Defaults to "sqlite".
    """
    # load study metadata
    # study_meta = load_study_metadata()
    pass
