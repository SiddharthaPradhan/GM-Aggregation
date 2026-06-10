"""Command-line interface for GM aggregation."""

import argparse
import logging
from .pipeline import run


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
    return vars(parser.parse_args())


def main() -> int:
    args = setup_parser()

    logging.getLogger().setLevel(logging.ERROR)
    logger = logging.getLogger("GM-Aggregator")
    logger.setLevel(logging.DEBUG if args["verbose"] else logging.INFO)
    logger.debug(args)

    run(
        input_path=args["input"],
        output_dir=args["output"],
        output_type=args["output_type"],
        n_jobs=args["njobs"],
        overwrite=args["overwrite"],
        verbose=args["verbose"],
        logger=logger,
    )

    return 0
