import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from collections import defaultdict

def data_partition(fname):
    df = pd.read_csv(fname)

    itemnum = df['item_id'].max()

    user_train = defaultdict(list)

    df = df.sort_values(by=['user_id', 'datetime'])
    for user_id, group in df.groupby('user_id'):
        items = group['item_id'].tolist()
        n_items = len(items)
        if n_items >= 3:
            user_train[user_id] = items[:-2]
        elif n_items == 2:
            user_train[user_id] = [items[0]]
        else:
            user_train[user_id] = items

    return [user_train, itemnum]

def obtain_cohorts(dataset):
    itemnum = dataset[1]
    freq_array = np.zeros(itemnum)
    for u_train in dataset[0].items():
        freq_array[[i - 1 for i in u_train[1]]] += 1

    total_downloads = freq_array.sum()

    cohorts = {"head":set(), "mid":set(), "tail":set()}
    cumul_freq = 0
    for item, freq in zip(list(np.argsort(-freq_array)), list(np.sort(-freq_array))):
        cumul_freq += -freq
        if cumul_freq < total_downloads / 3:
            cohorts["head"].add(item+1)
        elif cumul_freq < 2 *total_downloads / 3:
            cohorts["mid"].add(item+1)
        else:
            cohorts["tail"].add(item+1)

    cohorts["head_list"] = list(cohorts["head"])
    cohorts["mid_list"] = list(cohorts["mid"])
    cohorts["tail_list"] = list(cohorts["tail"])

    return cohorts

if __name__ == "__main__":
    # Based on https://github.com/apple/ml-negative-sampling/tree/main
    dataset = data_partition("../datasets/interactions.csv")
    cohorts = obtain_cohorts(dataset)
    print(len(cohorts["head"]), len(cohorts["mid"]), len(cohorts["tail"]))
    