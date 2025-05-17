import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline

plt.style.use('seaborn-v0_8')
sns.set_theme() 

datasets = {
    'ml_20m': "datasets/ratings.csv",
    'ml_1m': "datasets/ratings.dat",
    'kion': "datasets/interactions.csv"
}

fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("husl", len(datasets))


stats_data = []
for idx, (dataset_name, file_path) in enumerate(datasets.items()):
    print(f"Processing {dataset_name}...")
    df = pd.read_csv(file_path)
    item_interactions = df['item_id'].value_counts()
    item_interactions = item_interactions.sort_values(ascending=False)
    x = np.arange(len(item_interactions)) / (len(item_interactions) - 1)
    y = item_interactions.values
    y_normalized = y / y.sum()
    ax.plot(x, y_normalized, 
            label=dataset_name.upper(),
            color=colors[idx],
            alpha=0.3,
            linewidth=1)
    

    x_smooth = np.linspace(0, 1, 100)
    spl = make_interp_spline(x, y_normalized, k=3)
    y_smooth = spl(x_smooth)
    ax.plot(x_smooth, y_smooth, 
            color=colors[idx],
            linewidth=2)
    
    total_items = len(item_interactions)
    total_users = df['user_id'].nunique()
    total_interactions = item_interactions.sum()
    
    cumulative_freq = np.cumsum(y_normalized)
    head_idx = np.where(cumulative_freq >= 0.8)[0][0]  
    mid_idx = np.where(cumulative_freq >= 0.95)[0][0] 

    cohorts = [
        (0, head_idx, "Head"),
        (head_idx, mid_idx, "Mid"),
        (mid_idx, total_items, "Tail")
    ]
    
    for start, end, cohort_name in cohorts:
        cohort_items = item_interactions.iloc[start:end]
        num_items = len(cohort_items)
        items_percentage = (num_items / total_items) * 100
        item_user_counts = df[df['item_id'].isin(cohort_items.index)]['item_id'].value_counts()
        median_users = item_user_counts.median()
        mpu = (median_users / total_users) * 100
        interactions_percentage = (cohort_items.sum() / total_interactions) * 100
        
        stats_data.append({
            'Dataset': dataset_name.upper(),
            'Cohort': cohort_name,
            'Items (%)': f"{items_percentage:.1f}%",
            'MPU (%)': f"{mpu:.1f}%",
            'Interactions (%)': f"{interactions_percentage:.1f}%"
        })

stats_df = pd.DataFrame(stats_data)
print("\nPopularity Cohort Statistics:")
print(stats_df.to_string(index=False))
ax.set_yscale('log')  
ax.set_xlabel('Normalized Item Rank')
ax.set_ylabel('Normalized Item Frequency (log scale)')
ax.set_title('Item Popularity Distribution')
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.2)

stats_text = "Dataset Statistics:\n"
for dataset_name in datasets.keys():
    df = pd.read_csv(datasets[dataset_name])
    item_interactions = df['item_id'].value_counts()
    total_items = len(item_interactions)
    total_interactions = item_interactions.sum()
    stats_text += f"\n{dataset_name.upper()}:\n"
    stats_text += f"Total Items: {total_items:,}\n"
    stats_text += f"Total Interactions: {total_interactions:,}\n"
    stats_text += f"Avg Interactions/Item: {total_interactions/total_items:.1f}"
plt.tight_layout()
plt.savefig('item_popularity_distribution.png', dpi=300, bbox_inches='tight')