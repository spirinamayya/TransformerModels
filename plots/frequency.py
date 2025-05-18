import pandas as pd
import matplotlib.pyplot as plt

def frequency_distribution(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    item_freq = df['item_id'].value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(item_freq)), item_freq.values, color='steelblue')
    plt.yscale('log')  
    plt.xlabel('Items')
    plt.ylabel('Number of interactions')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    frequency_distribution("../Datasets/interactions.csv")
