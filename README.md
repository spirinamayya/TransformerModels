# Defining transformer model architecture
This repository contains source code for experiments conducted to determine optimal transformer architecture for recommender systems.

## Repository structure

### Datasets folder
Here datasets for experiment 1 are located, bigger datasets should be downloaded using following links:
- ML-1M-bert: https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/ml-1m.txt
- Beauty: https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/beauty.txt
- Steam: https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/steam.txt
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- MovieLens 20M: https://grouplens.org/datasets/movielens/20m/
- KION: https://github.com/irsafilo/KION_DATASET

### RecTools 
- models.
This folder contains code written for [RecTools](https://github.com/MobileTeleSystems/RecTools) library. To make experiments more convenient further it will be installed directly by calling ```pip install rectools[torch]```.
- examples.
Examples describe how to use transformer model developed, how to add item feature and shows figures from experiment 4.

### modifications
Contains modifications that were not released as a part of RecTools library.

### Validation
This notebook describes how experiments 2 and 3 were conducted.

### Results
Here reports for experiments 1, 2 and 3 can be found. Based on them insights about optimal transformer model were developed.
