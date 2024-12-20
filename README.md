# MLB Pitching Strategy Analysis

This repository provides tools for analyzing MLB pitching data, focusing on metrics such as `Pitching+`, `xRV`, and sequencing strategies. The project leverages Statcast data to develop data-driven insights into pitcher performance and in-game decisions.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Code Description](#code-description)
    - [Scraper](#scraper)
    - [Pitching+ Models](#pitching-models)
    - [Manager Models](#manager-models)
    - [Sequencing Models](#sequencing-models)
    - [Comparison Tools](#comparison-tools)
    - [Visual Tool](#visual-tool)
4. [Outputs](#outputs)
5. [Requirements](#requirements)

---

## Overview

This repository includes:

- **Data Scraping**: Download Statcast data for specific date ranges.
- **Pitching+ Analysis**: Calculate advanced metrics like `xRV` and aggregate pitcher statistics.
- **Sequencing Analysis**: Investigate how pitch sequences impact batter outcomes.
- **Managerial Decision Models**: Analyze in-game strategies and identify optimal decisions.
- **Visualization Tools**: Generate plots and visualizations for specific games and pitchers
- **Comparison Tools**: Compare models to external benchmarks like FanGraphs.

---

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com//alex-romero-ruffo/In-Game-Pitching-Thesis.git
    cd 'Thesis Final'
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the `scraper.py` script to download Statcast data for analysis.

---

## Code Description

### 1. Scraper

The `scraper.py` script downloads Statcast data for a specified date range and saves it as a CSV file.

**Example Usage:**
```bash
# Download Statcast data for 2022
python scraper.py --start_date 2022-04-07 --end_date 2022-11-03 --output_file statcast_2022.csv

# Download Statcast data for 2023
python scraper.py --start_date 2023-03-30 --end_date 2023-11-03 --output_file statcast_2023.csv
```

### 2. Pitching+ Models

The `Pitching+v2.ipynb` notebook calculates advanced metrics (`xRV`, `Pitching+`) and aggregates pitcher statistics. This relies on statcast and spin data, please ensure the data is placed in the appropriate locations.

**Inputs:**
- The Statcast data downloaded via `scraper.py`.

**Outputs:**
- `total_df`: Pitch-level data with features like `xRV` and other engineered metrics.
- `agg_total`: Aggregated statistics for each pitcher (e.g., averages across seasons).
- `stats_pitching`: Contains metrics like `wOBA`.

**Steps:**
1. Load the Statcast CSV files into the notebook.
2. Run all cells to generate the outputs.

### 3. Sequencing Models

The `Sequencing.ipynb` notebook analyzes pitch sequencing, exploring how different sequences impact outcomes like whiff rate, ground-ball rate, and run prevention.

**Inputs:**
- `total_df` generated from `Pitching+v2.ipynb`.

**Steps:**
1. Load the `total_df` DataFrame.
2. Run the notebook to perform sequencing analysis and generate insights.

### 4. Manager Models

The `Analysisv3.ipynb` notebook focuses on managerial decision-making, such as when to substitute pitchers or adjust strategies.

**Inputs:**
- `total_df` generated from `Pitching+v2.ipynb`.

**Steps:**
1. Load the `total_df` DataFrame.
2. Run the notebook to analyze managerial decisions and evaluate model outputs.

### 5. Comparison Tools

The comparison scripts allow you to benchmark your models against external sources like FanGraphs.

- `compare_model_with_fangraphs.py`: Compares `Pitching+` model outputs with FanGraphs' metrics.
- `compare_MSE.py`: Evaluates the mean squared error (MSE) of your models.

**Steps:**
1. Run these scripts after generating `agg_total` and `stats_pitching` in `Pitching+v2.ipynb`.

**Example:**
```bash
python compare_model_with_fangraphs.py
```

### 6. Visual Tool

The `visualization.py` script generates in-game pitcher stastics. this is not needed for the other models, as it is only a preliminary visual aid and a precursor to the models.


---

## Outputs

- **`total_df`**: Pitch-level dataset with engineered features like `xRV` and `Pitching+`.
- **`agg_total`**: Aggregated pitcher statistics across multiple seasons.
- **Plots and Graphs**: Visualizations generated by `visualization.py` and the notebooks.

---

## Requirements

The project requires Python 3.12+ and the packages listed in `requirements.txt`. Install dependencies using:

```bash
pip install -r requirements.txt
```

Key packages include:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `xgboost`
- `catboost`
- `pybaseball`
- `optuna`

For additional dependencies, see `requirements.txt`.

---

## Notes

- Ensure proper caching with `pybaseball` by enabling it in your Python scripts (`cache.enable()`).
- Modify date ranges and model parameters to suit your analysis needs.
- Outputs can be further visualized or exported for external reporting.

---
```
