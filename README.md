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
python main.py --help
```

## Usage

### Basic Usage

```bash
# If using uv:
uv run main.py --input /path/to/data --output-type csv

# If using venv:
python main.py --input /path/to/data --output-type csv
```

### Command Line Arguments

-   `--input, -i`: **Required.** Path to input directory or zip file containing GMA data
-   `--output-type, -t`: **Required.** Output format: `csv`, `sqlite`, or `both`
-   `--output, -o`: Output directory (default: `./output/`)
-   `--njobs, -j`: Number of parallel jobs for aggregation (default: -1, uses all cores)
-   `--overwrite, -w`: Overwrite existing output directory if it exists
-   `--verbose, -v`: Enable verbose logging for debugging

### Examples

```bash
# Process a zip file and save as CSV
python main.py -i data/study-data.zip -t csv -o ./results/

# Process a directory with SQLite output and parallel processing
python main.py -i /path/to/data/ -t sqlite -j 4 -v

# Process with both CSV and SQLite output, overwrite existing files
python main.py -i data.zip -t both -w
```

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

-   `main.py`: Command-line interface and orchestration
-   `preprocess.py`: Data cleaning and preprocessing functions
-   `aggregate.py`: Multi-level aggregation logic
-   `utils.py`: Utility functions for file I/O and data handling

## Development

### Project Structure

```
GM-Aggregation/
├── main.py              # CLI entry point
├── preprocess.py        # Data preprocessing
├── aggregate.py         # Aggregation logic
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
├── pyproject.toml     # Project configuration
└── README.md          # Documentation
```

## License

This project is part of educational research. Please contact the maintainer for usage permissions.
