import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Dataset
# This already loads the full file into memory
df = pd.read_csv('Global YouTube Statistics.csv')

# Step 2: Inspect the Data
print("--- Dataset Information (Original) ---")
df.info()
print("-" * 40)

# --- MODIFICATION 1 ---
# Instead of dropping rows, let's automatically select all
# numerical columns and fill their missing values with 0.
# This keeps all the rows from the "full file."

# Select all columns that are numbers (int64 or float64)
numerical_cols_df = df.select_dtypes(include=['int64', 'float64'])

# Fill missing data in *only* these numerical columns with 0
numerical_cols_df = numerical_cols_df.fillna(0)

print(f"Successfully found {len(numerical_cols_df.columns)} numerical columns to analyze.")


# Step 3: Correlation Analysis
# --- MODIFICATION 2 ---
# We now create the correlation matrix from our new DataFrame
# which includes *all* numerical columns.
correlation_matrix = numerical_cols_df.corr()

print("\n--- Correlation Matrix (All Numerical Columns) ---")
print(correlation_matrix)
print("-" * 40)


# Step 4: Visualize the Correlation Matrix with a Heatmap
# This plot might be much larger now, as it shows all columns
plt.figure(figsize=(12, 10)) # Made figure larger for more columns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
plt.title('Correlation Matrix of *All* YouTube Metrics', fontsize=16)
plt.savefig('youtube_heatmap_all_columns.png')
plt.show()


# Step 5: Visualize Specific Relationships with Scatter Plots
# --- MODIFICATION 3 ---
# The original code plotted 'Views' vs 'Likes'.
# Your file might have different names (e.g., 'video views').
# This "try...except" block will try to make the plot,
# but will just print a warning if it fails, so the code doesn't crash.

# --- IMPORTANT: UPDATE THESE NAMES ---
# Check your CSV and update these two variable names to match!
x_axis_column = 'Views'
y_axis_column = 'Likes'

try:
    plt.figure(figsize=(10, 5))
    # We use 'df' here, not 'numerical_cols_df', as scatterplot handles missing data well
    sns.scatterplot(data=df, x=x_axis_column, y=y_axis_column, alpha=0.5)
    plt.title(f'{x_axis_column} vs. {y_axis_column}', fontsize=16)
    plt.xlabel(f'Total {x_axis_column}', fontsize=12)
    plt.ylabel(f'Total {y_axis_column}', fontsize=12)
    plt.grid(True)
    plt.savefig('youtube_views_vs_likes.png')
    plt.show()

except KeyError:
    print(f"\nWarning: Could not create scatter plot.")
    print(f"One or both columns ('{x_axis_column}', '{y_axis_column}') were not found in your CSV.")
    print("Please check the column names printed by 'df.info()' and update Step 5.")