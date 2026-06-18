# GMA Data Aggregation Tool

A Python tool for preprocessing and aggregating Graspable Math Acitivities (GMA) data from research studies.

## Installation

### Requirements

-   Python 13.0 or higher

### Setup

```bash
# Clone the repository
git clone https://github.com/SiddharthaPradhan/GM-Aggregation.git
cd GM-Aggregation
```

### Method 1: Using uv (recommended)

```bash
# Create virtual environment and install dependencies with uv
uv sync
```

### Method 2: Using pip with virtual environment

```bash
# Create and activate virtual environment
python -m venv

# Activate virtual environment
# On Linux/macOS
source .venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test the installation
uv run python -m gm_aggregation --help
```

## Usage

### Basic Usage

```bash
# If using uv:
uv run python -m gm_aggregation --input /path/to/data --output-type csv

# If using venv:
python -m gm_aggregation --input /path/to/data --output-type csv

# If installed as a package, use the console script:
gm-aggregation --input /path/to/data --output-type csv
```

### Command Line Arguments

-   `--input, -i`: **Required.** Path to input directory or zip file containing GMA data
-   `--output-type, -t`: **Required.** Output format: `csv`, `sqlite`, or `both`
-   `--output, -o`: Output directory (default: `./output/`)
-   `--njobs, -j`: Number of parallel jobs for aggregation (default: -1, uses all cores)
-   `--overwrite, -w`: Overwrite existing output directory if it exists
-   `--verbose, -v`: Enable verbose logging for debugging
-   `--dask`: Enable Dask distributed processing for memory-constrained environments
-   `--memory-limit`: Total Dask memory budget across all workers (e.g., `800MB`, `1GB`). Default: `6GB`. Excess is spilled to disk.

### Examples

```bash
# Process a zip file and save as CSV
python -m gm_aggregation -i data/study-data.zip -t csv -o ./results/

# Process a directory with SQLite output and parallel processing
python -m gm_aggregation -i /path/to/data/ -t sqlite -j 4 -v

# Process with both CSV and SQLite output, overwrite existing files
python -m gm_aggregation -i data.zip -t both -w

# Process with Dask for memory-constrained environments (e.g., Heroku)
python -m gm_aggregation -i data.zip -t csv --dask --memory-limit 800MB

# Process with Dask and specific worker configuration
python -m gm_aggregation -i data.zip -t csv --dask --memory-limit 1GB -j 2
```

## Parallel Processing

### Multiprocessing (Default)
By default, the tool uses Python's multiprocessing module with the specified number of jobs (`-j`). This approach is suitable for most use cases and keeps memory usage predictable.

### Dask Distributed Processing
For memory-constrained environments (e.g., Heroku dynos), use the `--dask` flag:

```bash
python -m gm_aggregation -i data.zip -t csv --dask --memory-limit 800MB
```

**Key features:**
- **Memory management**: Dask respects the memory limit and automatically spills excess to disk
- **Streaming concatenation**: Results are written to disk incrementally, with chunked reading to prevent unbounded memory accumulation
- **Spill-to-disk**: Temporary spill directory is used for overflow and cleaned up after processing

**When to use:**
- Processing on Heroku or other memory-constrained cloud platforms
- Working with large datasets that approach available RAM
- When `MemoryError` occurs with the default multiprocessing approach

## Input Data Structure

The tool expects GMA research data export containing these files:

### Required Files

-   `study-metadata.json`: Study-level information and configuration
-   `task-metadata.json`: Task and problem definitions
-   `event-logs.json`: Detailed user interaction events
-   `attempt-data.json`: Student attempt summaries
-   `roster-metadata.json`: Student roster (optional for public sessions)

### Input Formats

1. **Zip File**: All required JSON files in a single zip archive
2. **Directory**: Folder containing all required JSON files

## Output Structure

```
output/
└── {study-id}/
    ├── study_info.txt           # Human-readable study metadata
    ├── event_logs.csv           # Preprocessed event data
    ├── roster_metadata.csv      # Student roster (if available)
    ├── task_metadata.csv        # Task definitions
    └── GMA_data.db             # SQLite database (if selected)
```

## File Descriptions

-   `src/gm_aggregation/cli.py`: Command-line interface and orchestration
-   `src/gm_aggregation/preprocess.py`: Data cleaning and preprocessing functions
-   `src/gm_aggregation/aggregate.py`: Multi-level aggregation logic
-   `src/gm_aggregation/utils.py`: Utility functions for file I/O and data handling

## Development

### Project Structure

```
GM-Aggregation/
├── src/
│   └── gm_aggregation/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── preprocess.py
│       ├── aggregate.py
│       ├── generate_graph.py
│       ├── generate_classifications.py
│       └── utils.py
├── requirements.txt    # Dependencies
├── pyproject.toml     # Project configuration
└── README.md          # Documentation
```

## License

This project is part of educational research. Please contact the maintainer for usage permissions.
