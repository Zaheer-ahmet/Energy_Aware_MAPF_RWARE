import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# ---------------------------
# 1. READ AND PREPARE DATA
# ---------------------------
file_path = "/Users/ugurendirlik/Desktop/ITU ders projeleri/AI Proje MAPF/MAPF_Experiment_Configuration_Matrix - MAPF_Experiment_Configuration_Matrix.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=['Layout', 'RunID', 'Command'], errors='ignore')
print(df.head())
cols = ['Agents', 'Obstacles(%)', 'Chargers', 'Battery',
        'Total Accumulated Team Reward', 'Total Steps Taken']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter for only Battery == 50 or Battery == NaN
df = df[(df['Battery'] == 50) | (df['Battery'].isna())].copy()

# Label battery mode
df['Battery Mode'] = df['Battery'].apply(lambda x: 'No Battery' if pd.isna(x) else 'Battery=50')

# ---------------------------
# 2. NORMALIZED LINE PLOTS
# ---------------------------
def normalize_and_plot(filtered_df, group_col, title, xlabel):
    grouped = filtered_df.groupby(group_col)[['Total Accumulated Team Reward', 'Total Steps Taken']].mean().reset_index()
    scaler = MinMaxScaler()
    grouped[['Normalized Reward', 'Normalized Steps']] = scaler.fit_transform(
        grouped[['Total Accumulated Team Reward', 'Total Steps Taken']]
    )
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=grouped, x=group_col, y='Normalized Reward', marker='o', label='Normalized Team Reward')
    sns.lineplot(data=grouped, x=group_col, y='Normalized Steps', marker='s', label='Normalized Steps Taken')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Value (0-1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot 1: Agent Count Effect (Obstacle=10%, Battery=50)
normalize_and_plot(df[(df['Obstacles(%)'] == 10) & (df['Battery'] == 50)],
                   'Agents', "Effect of Agent Count (Obstacle=10%, Battery=50)", "Number of Agents")

# Plot 2: Obstacle Density Effect (Agents=4, Battery=50)
normalize_and_plot(df[(df['Agents'] == 4) & (df['Battery'] == 50)],
                   'Obstacles(%)', "Effect of Obstacle Density (Agents=4, Battery=50)", "Obstacle Density (%)")

# Plot 3: (Battery level not plotted here because we fix Battery=50 or NaN)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load CSV file
file_path = "/Users/ugurendirlik/Desktop/ITU ders projeleri/AI Proje MAPF/MAPF_Experiment_Configuration_Matrix - MAPF_Experiment_Configuration_Matrix.csv"
df = pd.read_csv(file_path)

# Convert numeric columns
cols = ['Agents', 'Obstacles(%)', 'Chargers', 'Battery',
        'Total Accumulated Team Reward', 'Total Steps Taken']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Battery mode labels based on battery values
def battery_label(x):
    if pd.isna(x):
        return 'No Battery'
    elif x == 15:
        return 'Battery=15'
    elif x == 25:
        return 'Battery=25'
    elif x == 50:
        return 'Battery=50'
    else:
        return 'Other'

df['Battery Mode'] = df['Battery'].apply(battery_label)

# Only desired battery modes: 15, 25, 50, No Battery
scatter_df = df[df['Battery Mode'].isin(['Battery=15', 'Battery=25', 'Battery=50', 'No Battery'])].copy()

# Remove missing values
scatter_df = scatter_df.dropna(subset=['Total Accumulated Team Reward', 'Total Steps Taken'])

# Normalization
scaler = MinMaxScaler()
scatter_df[['Normalized Reward', 'Normalized Steps']] = scaler.fit_transform(
    scatter_df[['Total Accumulated Team Reward', 'Total Steps Taken']]
)

# Scatter plot drawing
plt.figure(figsize=(9, 6))
palette = {'Battery=15': 'blue', 'Battery=25': 'green', 'Battery=50': 'red', 'No Battery': 'orange'}

sns.scatterplot(
    data=scatter_df,
    x='Normalized Reward',
    y='Normalized Steps',
    hue='Battery Mode',
    palette=palette,
    s=80,
    alpha=0.8
)

plt.title("Battery Level Comparison: Normalized Reward vs Steps")
plt.xlabel("Normalized Total Accumulated Team Reward")
plt.ylabel("Normalized Total Steps Taken")
plt.legend(title='Battery Mode')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# 4. LINEAR REGRESSION 
# ---------------------------
df_battery = df.dropna()
X = df_battery[['Agents', 'Obstacles(%)', 'Chargers', 'Battery']]
y_reward = df_battery['Total Accumulated Team Reward']
y_steps = df_battery['Total Steps Taken']

model_reward = LinearRegression().fit(X, y_reward)
model_steps = LinearRegression().fit(X, y_steps)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Impact on Reward': model_reward.coef_,
    'Impact on Steps': model_steps.coef_
})
importance_df['Abs Impact on Reward'] = importance_df['Impact on Reward'].abs()
importance_df['Abs Impact on Steps'] = importance_df['Impact on Steps'].abs()
importance_df_sorted = importance_df.sort_values(by='Abs Impact on Reward', ascending=False)

print("\nImpact of Input Variables on Reward and Steps:\n")
print(importance_df_sorted[['Feature', 'Impact on Reward', 'Impact on Steps']])
