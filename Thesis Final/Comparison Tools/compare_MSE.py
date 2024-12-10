import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the two CSV files into DataFrames
print("Loading CSV files...")
df1 = pd.read_csv('agg_total.csv')  
df2 = pd.read_csv('fangraphs-leaderboards_20IP_2022-2023.csv')
print("Files loaded successfully.")
print(f"Initial row count for agg_total: {len(df1)}")
print(f"Initial row count for fangraphs-leaderboards: {len(df2)}")

# Clean and standardize name columns
print("Standardizing name formats...")
df1[['Last Name', 'First Name']] = df1['player_name'].str.split(',', expand=True)
df1['First Name'] = df1['First Name'].str.strip()
df1['Last Name'] = df1['Last Name'].str.strip()
df1['Full Name'] = df1['First Name'] + ' ' + df1['Last Name']

df2['NameASCII'] = df2['NameASCII'].str.strip()

# Merge on the standardized name column
print("Merging DataFrames on standardized name columns...")
merged_df = pd.merge(df1, df2, left_on='Full Name', right_on='NameASCII', suffixes=('_agg', '_fangraphs'))

# Convert 'Pitching+' columns to numeric if not already
print("Converting Pitching+ columns to numeric...")
merged_df['Pitching+'] = pd.to_numeric(merged_df['Pitching+'], errors='coerce')
merged_df['pitching_plus'] = pd.to_numeric(merged_df['pitching_plus'], errors='coerce')
merged_df = merged_df.dropna(subset=['Pitching+', 'pitching_plus'])
print(f"Row count after dropping missing Pitching+ values: {len(merged_df)}")

# Calculate ranks based on Pitching+
print("Calculating ranks based on Pitching+ values...")
merged_df['rank_fangraphs'] = merged_df['Pitching+'].rank(ascending=False)
merged_df['rank_agg'] = merged_df['pitching_plus'].rank(ascending=False)

# Calculate the rank difference
merged_df['rank_difference'] = abs(merged_df['rank_fangraphs'] - merged_df['rank_agg'])

# Calculate the difference in Pitching+ values
merged_df['pitching_plus_difference'] = merged_df['Pitching+'] - merged_df['pitching_plus']

# Calculate metrics
print("\nCalculating similarity metrics...")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(merged_df['Pitching+'], merged_df['pitching_plus'])
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(merged_df['Pitching+'], merged_df['pitching_plus'])
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Correlation Coefficient
correlation = merged_df[['Pitching+', 'pitching_plus']].corr().iloc[0, 1]
print(f"Pearson Correlation Coefficient: {correlation:.4f}")

# Spearman Rank Correlation
spearman_rank_corr = merged_df[['rank_fangraphs', 'rank_agg']].corr(method='spearman').iloc[0, 1]
print(f"Spearman Rank Correlation: {spearman_rank_corr:.4f}")

# Top 20 overlap
top_n = 20
top_fangraphs = set(merged_df.nsmallest(top_n, 'rank_fangraphs')['Full Name'])
top_agg = set(merged_df.nsmallest(top_n, 'rank_agg')['Full Name'])
top_n_overlap = len(top_fangraphs & top_agg)
print(f"Top-{top_n} Consistency: {top_n_overlap}/{top_n} ({top_n_overlap / top_n * 100:.2f}%)")

# Save to CSV
output_file = 'pitching_plus_comparison_debug.csv'
merged_df.to_csv(output_file, index=False)
print(f"Detailed comparison saved to '{output_file}'")

# Calculate correlation between Fangraphs Pitching+ and ERA
print("\nCalculating Fangraphs Pitching+ and ERA correlation...")

merged_df['ERA'] = pd.to_numeric(merged_df['ERA'], errors='coerce')
merged_df = merged_df.dropna(subset=['ERA'])

# Pearson Correlation between Pitching+ and ERA
era_pitching_corr = merged_df[['Pitching+', 'ERA']].corr().iloc[0, 1]
print(f"Pearson Correlation between Fangraphs Pitching+ and ERA: {era_pitching_corr:.4f}")

# Spearman Correlation between Pitching+ and ERA
era_pitching_spearman_corr = merged_df[['Pitching+', 'ERA']].corr(method='spearman').iloc[0, 1]
print(f"Spearman Rank Correlation between Fangraphs Pitching+ and ERA: {era_pitching_spearman_corr:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['pitching_plus'], merged_df['Pitching+'], alpha=0.7, label='Data Points')

# Perfect correlation line (y = x)
min_val = min(merged_df['pitching_plus'].min(), merged_df['Pitching+'].min())
max_val = max(merged_df['pitching_plus'].max(), merged_df['Pitching+'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='Perfect Correlation (y=x)')

# Actual correlation line (regression line)
X = merged_df['pitching_plus'].values.reshape(-1, 1)
y = merged_df['Pitching+'].values
reg_model = LinearRegression()
reg_model.fit(X, y)
y_pred = reg_model.predict([[min_val], [max_val]])
plt.plot([min_val, max_val], y_pred, color='blue', label=f'Actual Correlation (y={reg_model.coef_[0]:.2f}x+{reg_model.intercept_:.2f})')

# Labels and title
plt.xlabel('Model Pitching+')
plt.ylabel('Fangraphs Pitching+')
plt.title('Comparison of Pitching+ Metrics')
plt.legend()
plt.grid()
plt.show()