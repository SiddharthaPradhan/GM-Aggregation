"""
Handles CMD line input
"""

import argparse
import os
import logging
from aggregate import aggregate_and_save
from preprocess import (
    preprocess_and_save_metadata,
    preprocess_and_save_event_log,
    save_study_metadata,
)
from utils import CONTENT_DICT, check_existence, load_zip, get_study_id
import sys


def setup_parser() -> dict:
    parser = argparse.ArgumentParser(description="Aggregate GMA data.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory or zip containing GMA data files",
    )
    parser.add_argument(
        "--output-type",
        "-t",
        type=str,
        required=True,
        choices=["sqlite", "csv", "both"],
        help="Type of output file",
    )
    parser.add_argument(
        "--njobs",
        "-j",
        type=int,
        help="Number of parallel jobs to use for aggregation (each task/problem is assigned to a thread)",
        default=-1,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for aggregated GMA data",
        default="./output/",
    )

    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="Overwrite contents in output directory if it exists",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = vars(parser.parse_args())
    return args


def validate_and_process_args(args: dict, logger: logging.Logger):
    """Validate command line arguments and prepare input/output paths.

    Args:
        args: Parsed command line arguments
        logger: Logger instance for logging messages

    Returns:
        tuple: (input_data, study_output_dir) or (None, None) if validation fails
    """
    # check if input path exists
    check_existence(args["input"])
    logger.debug(f"Input path verified: {args['input']}")

    # create output directory if it doesn't exist
    os.makedirs(args["output"], exist_ok=True)
    logger.debug(f"Output directory created/verified: {args['output']}")

    # get input file from the input path, str if directory else ZipFile
    input_file = None
    # get study name to create output sub-directory
    study_meta_file = None
    if args["input"].endswith(".zip"):
        input_file = load_zip(args["input"])
        study_meta_file = input_file.open(CONTENT_DICT["study"])
    else:
        input_file = args["input"]
        study_meta_file = open(os.path.join(input_file, CONTENT_DICT["study"]), "r")

    study_name = get_study_id(study_meta_file)
    args["study_output_dir"] = os.path.join(args["output"], study_name)
    args["input_file"] = input_file
    logger.debug(f"Output will be stored in: {args['study_output_dir']}")

    # raise error if sub-directory already exists
    if os.path.exists(args["study_output_dir"]):
        if args.get("overwrite", False):
            logger.warning(
                f"Output directory already exists: {args['study_output_dir']}. "
                "Contents may be overwritten."
            )
        else:
            raise FileExistsError(
                f"Output directory already exists: {args['study_output_dir']}. "
                "Remove it or choose a different output directory to avoid overwriting data."
            )
    else:  # create output sub-directory
        os.makedirs(args["study_output_dir"], exist_ok=True)
        logger.info(f"Output will be stored in: {args['study_output_dir']}")

    return args


def main():
    args = setup_parser()
    logging.basicConfig(level=logging.DEBUG if args["verbose"] else logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting GMA data preprocessing and aggregation")

    logger.debug(args)

    # Validate arguments and prepare paths
    args = validate_and_process_args(args, logger)

    # preprocess and save metadata
    preprocess_and_save_metadata(
        args["input_file"], args["study_output_dir"], args["output_type"]
    )
    # preprocess event log and save
    event_log_df = preprocess_and_save_event_log(
        args["input_file"], args["study_output_dir"], args["output_type"]
    )
    aggregate_and_save(
        event_log_df,
        args["study_output_dir"],
        args["output_type"],
        n_jobs=args["njobs"],
    )

    # exit
    logger.info(f"All processed files saved to {args['study_output_dir']}")
    logger.info("Finished..")
    return 0


if __name__ == "__main__":
    main()
