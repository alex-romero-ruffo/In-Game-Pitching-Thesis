import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load the first CSV file containing Pitching+ data
pitching_plus_df = pd.read_csv(r'Pitching+ Models\stats_pitching.csv')

# Load the second CSV file containing traditional stats
traditional_stats_df = pd.read_csv('fangraphs-leaderboards_20IP_2022-2023.csv')

# Display the first few rows to check the data
print("Pitching+ Data:")
print(pitching_plus_df.head())
print("\nTraditional Stats Data:")
print(traditional_stats_df.head())

# Standardize player names in both DataFrames
def standardize_Name(name):
    # Remove any leading/trailing whitespace and convert to uppercase
    return name.strip().upper()

pitching_plus_df['Name'] = pitching_plus_df['Name'].apply(standardize_Name)
traditional_stats_df['Name'] = traditional_stats_df['Name'].apply(standardize_Name)

# Rename columns in traditional_stats_df to prevent conflicts
traditional_stats_df.rename(columns={
    'ERA': 'ERA_trad',
    'K/9': 'K9_trad',
    'BABIP': 'BABIP_trad',
    'K%': 'Kperc_trad',
    'WHIP': 'WHIP_trad',
    'FIP' : 'FIP_trad'
}, inplace=True)

# Merge the DataFrames on player names
merged_df = pd.merge(pitching_plus_df, traditional_stats_df, on='Name', how='inner')

print(f"\nNumber of players after merging: {len(merged_df)}")

# Print the columns to check for suffixes
print("Columns in merged_df after merging:")
print(merged_df.columns.tolist())

# Ensure relevant columns are numeric
columns_to_convert = ['pitching_plus', 'ERA_trad', 'K9_trad', 'BABIP_trad', 'Kperc_trad', 'WHIP_trad', 'FIP_trad']
for col in columns_to_convert:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Drop rows with missing values in the relevant columns
merged_df = merged_df.dropna(subset=columns_to_convert)
print(f"Number of players after dropping missing values: {len(merged_df)}")

# Calculate Pearson and Spearman correlations
correlation_results = {}
for metric in ['ERA_trad', 'K9_trad', 'BABIP_trad', 'Kperc_trad', 'WHIP_trad', 'FIP_trad']:
    pearson_corr, _ = pearsonr(merged_df['pitching_plus'], merged_df[metric])
    spearman_corr, _ = spearmanr(merged_df['pitching_plus'], merged_df[metric])
    correlation_results[metric] = {
        'Pearson Correlation': pearson_corr,
        'Spearman Correlation': spearman_corr
    }

# Print the results
print("\nCorrelation Results between Pitching+ and Traditional Stats:")
for metric, values in correlation_results.items():
    print(f"{metric}:\n  Pearson Correlation = {values['Pearson Correlation']:.4f}\n  Spearman Correlation = {values['Spearman Correlation']:.4f}\n")

output_file = 'pitching_plus_correlation_results.csv'
results_df = pd.DataFrame.from_dict(correlation_results, orient='index')
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Metric'}, inplace=True)
results_df.to_csv(output_file, index=False)
print(f"Detailed correlation results saved to '{output_file}'")

print('with fangraphs')
print('with fangraphs')

correlation_results = {}
for metric in ['ERA_trad', 'K9_trad', 'BABIP_trad', 'Kperc_trad', 'WHIP_trad', 'FIP_trad']:
    pearson_corr, _ = pearsonr(merged_df['Pitching+'], merged_df[metric])
    spearman_corr, _ = spearmanr(merged_df['Pitching+'], merged_df[metric])
    correlation_results[metric] = {
        'Pearson Correlation': pearson_corr,
        'Spearman Correlation': spearman_corr
    }

print("\nCorrelation Results between Pitching+ and Traditional Stats:")
for metric, values in correlation_results.items():
    print(f"{metric}:\n  Pearson Correlation = {values['Pearson Correlation']:.4f}\n  Spearman Correlation = {values['Spearman Correlation']:.4f}\n")

output_file = 'pitching_plus_correlation_results_with_fangraphs.csv'
results_df = pd.DataFrame.from_dict(correlation_results, orient='index')
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Metric'}, inplace=True)
results_df.to_csv(output_file, index=False)
print(f"Detailed correlation results saved to '{output_file}'")
