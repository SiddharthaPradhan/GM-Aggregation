"""Reusable pipeline orchestration for GM aggregation."""

from pathlib import Path
import logging
import os
import shutil
import tempfile
from typing import Callable, TypedDict

from .aggregate import aggregate_and_save
from .preprocess import preprocess_and_save_event_log, preprocess_and_save_metadata
from .utils import CONTENT_DICT, OUTPUT_TYPE, check_existence, get_study_id, load_zip


class ProgressEvent(TypedDict, total=False):
    stage: str
    completed: int
    total: int
    message: str


ProgressCallback = Callable[[ProgressEvent], None]


def run(
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike = "./output",
    output_type: OUTPUT_TYPE = "csv",
    convert_latex: bool = False,
    n_jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
    logger: logging.Logger | None = None,
    progress_callback: ProgressCallback | None = None,
    memory_limit: str = "800MB",
    use_dask: bool = True,
) -> tuple[str, Path]:
    """Run preprocessing + aggregation pipeline and return (study_id, output_dir)."""
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
        n_jobs = n_jobs - 1 if n_jobs > 1 else 1
    input_path = str(input_path)
    output_root = Path(output_dir)

    check_existence(input_path)
    output_root.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logging.getLogger().setLevel(logging.ERROR)
        logger = logging.getLogger("GM-Aggregator")
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.info("Starting GMA data preprocessing and aggregation")
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "starting",
                "message": "Starting pipeline",
            }
        )

    if input_path.endswith(".zip"):
        input_file = load_zip(input_path)
        with input_file.open(CONTENT_DICT["study"]) as study_meta_file:
            study_id = get_study_id(study_meta_file)
    else:
        input_file = input_path
        with open(
            os.path.join(input_path, CONTENT_DICT["study"]), "r"
        ) as study_meta_file:
            study_id = get_study_id(study_meta_file)

    study_output_dir = output_root / study_id
    logger.debug(f"Output will be stored in: {study_output_dir}")

    if study_output_dir.exists():
        if overwrite:
            logger.warning(
                f"Output directory already exists: {study_output_dir}. "
                "Contents may be overwritten."
            )
        else:
            raise FileExistsError(
                f"Output directory already exists: {study_output_dir}. "
                "Remove it or choose a different output directory to avoid overwriting data."
            )
    else:
        study_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be stored in: {study_output_dir}")

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "metadata_preprocessing",
                "message": "Starting metadata preprocessing",
            }
        )

    task_meta_df = preprocess_and_save_metadata(
        input_file, str(study_output_dir), output_type, convert_latex
    )
    if task_meta_df is None:
        raise RuntimeError("Task metadata preprocessing returned no data")

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "metadata_preprocessing",
                "message": "Finished metadata preprocessing",
            }
        )

    def _run_stages(dask_client) -> None:
        event_log_df = preprocess_and_save_event_log(
            input_file,
            str(study_output_dir),
            output_type,
            convert_latex=convert_latex,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
            dask_client=dask_client,
        )
        aggregate_and_save(
            event_log_df,
            task_meta_df,
            str(study_output_dir),
            output_type,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
            dask_client=dask_client,
        )

    if use_dask:
        n_workers = max(1, n_jobs)
        spill_dir = tempfile.mkdtemp(prefix="dask-spill-")
        try:
            from dask.distributed import Client, LocalCluster
            from dask.utils import parse_bytes

            per_worker_bytes = parse_bytes(memory_limit) // n_workers
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
                memory_limit=per_worker_bytes,
                local_directory=spill_dir,
                silence_logs=logging.WARNING,
            )
            logger.info(
                f"Dask cluster: {n_workers} worker(s), "
                f"{memory_limit} total memory limit, spill → {spill_dir}"
            )

            with Client(cluster) as dask_client:
                _run_stages(dask_client)
        finally:
            shutil.rmtree(spill_dir, ignore_errors=True)
    else:
        logger.info("Dask disabled — using pandas/multiprocessing path")
        _run_stages(dask_client=None)

    logger.info(f"All processed files saved to {study_output_dir}")
    logger.info("Finished..")
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "finished",
                "message": "Pipeline finished",
            }
        )
    return study_id, study_output_dir
