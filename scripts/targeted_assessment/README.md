# Targeted Assessment Module

This module contains the code for running behavioral assessments on language models using the AnaphoraGym benchmark.

## Structure

The module is organized into clear, separate components:

```
targeted_assessment/
├── experiments/          # Running experiments and tests
│   └── run_experiment.py
├── analysis/            # Data analysis and statistics
│   ├── aggregate_results.py
│   ├── compare_model_types.py
│   ├── compare_tuning_pairs.py
│   └── dataset_stats.py
├── visualization/       # Plotting and chart generation
│   ├── create_bar_chart.py
│   ├── create_faceted_chart.py
│   ├── create_heatmap.py
│   ├── create_radar_chart.py
│   └── create_comparison_charts.py
├── data/               # Data processing and consolidation
│   ├── concatenate_results.py
│   └── create_enriched_dataset.py
├── utils/              # Shared utilities
│   ├── paths.py
│   ├── data_loader.py
│   └── model_loader.py
└── run_all.sh          # Main orchestration script
```

## Usage

### Running Experiments

Run a single model experiment:
```bash
python scripts/targeted_assessment/experiments/run_experiment.py --model "gpt2"
```

Run all experiments (configured in `run_all.sh`):
```bash
bash scripts/targeted_assessment/run_all.sh
```

### Analysis

Aggregate results from all models:
```bash
python scripts/targeted_assessment/analysis/aggregate_results.py
```

Compare model types:
```bash
python scripts/targeted_assessment/analysis/compare_model_types.py
```

Get dataset statistics:
```bash
python scripts/targeted_assessment/analysis/dataset_stats.py
```

### Visualization

Create a bar chart:
```bash
python scripts/targeted_assessment/visualization/create_bar_chart.py
```

Create all comparison charts:
```bash
python scripts/targeted_assessment/visualization/create_comparison_charts.py
```

### Data Processing

Concatenate all results:
```bash
python scripts/targeted_assessment/data/concatenate_results.py
```

Create enriched dataset:
```bash
python scripts/targeted_assessment/data/create_enriched_dataset.py
```

## Key Features

- **Separation of Concerns**: Experiments, analysis, visualization, and data processing are clearly separated
- **Shared Utilities**: Common functionality (paths, data loading, model loading) is centralized
- **Modular Design**: Each script has a single, clear purpose
- **Easy to Extend**: Add new analyses or visualizations without modifying existing code

## Dependencies

All scripts use the shared utilities module (`utils/`) for:
- Path management
- Data loading
- Model loading

Make sure to run scripts from the project root directory for proper path resolution.

