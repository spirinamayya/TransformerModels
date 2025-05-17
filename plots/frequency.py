import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
sns.set_theme()

datasets = {
    'ml_20m': "datasets/ratings.csv",
    'ml_1m': "datasets/ratings.dat",
    'kion': "datasets/interactions.csv"
}

fig, axes = plt.subplots(3, 1, figsize=(15, 18))
fig.suptitle('Frequency Distribution of User-Item Interactions', fontsize=16, y=0.95)

for idx, (dataset_name, file_path) in enumerate(datasets.items()):
    print(f"Processing {dataset_name}...")
    df = pd.read_csv(file_path)
    user_interactions = df['user_id'].value_counts()
    bins = pd.qcut(user_interactions, q=200, duplicates='drop')
    bin_counts = bins.value_counts().sort_index()
    ax = axes[idx]
    bars = ax.bar(range(len(bin_counts)), bin_counts.values, 
                 color=sns.color_palette("husl", len(bin_counts)))
    ax.set_title(f'{dataset_name.upper()} Dataset', pad=20)
    ax.set_xlabel('Interaction Count Range')
    ax.set_ylabel('Number of Users')
    ax.set_yscale('log')  
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    bin_labels = [f"{int(bin.left)}-{int(bin.right)}" for bin in bin_counts.index]
    ax.set_xticks(range(len(bin_counts)))
    ax.set_xticklabels(bin_labels)
    stats_text = f'Mean: {user_interactions.mean():.2f}\n'
    stats_text += f'Median: {user_interactions.median():.2f}\n'
    stats_text += f'Max: {user_interactions.max()}\n'
    stats_text += f'Min: {user_interactions.min()}'
    
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('frequency_distribution.png', dpi=300, bbox_inches='tight')
