import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from classification.labels import unique_labels


def plot_label_counts(df):
    """Creates bar plot of number of songs per artist

    @param df:                                  dataframe of dataset
    """
    counts = [df[df['artist'] == a].shape[0] for a in unique_labels]
    labels = [l[2:-2] for l in unique_labels]

    plt.bar(labels, counts)
    plt.xticks(rotation=-30)
    plt.show()


def plot_length_hist(df):
    """Creates histogram of line lengths in ms

    @param df:                                  dataframe of dataset
    """
    lengths = df['length'].values

    plt.hist(lengths, 50)
    mean = plt.axvline(np.mean(lengths), color='red')
    plt.legend([mean], [f'Mean: {np.round(np.mean(lengths), 2)}'])
    plt.show()


def main(train_csv, val_csv):
    """Print dataset statistics

    @param train_csv:                           path to train .csv file
    @param val_csv:                             path to val .csv file
    """
    df = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)])

    print(f'{"Number of samples:".ljust(50)}{df.shape[0]}')

    plot_label_counts(df)
    plot_length_hist(df)


if __name__ == '__main__':
    main(train_csv=os.path.join('..', 'dataset', 'train_val_data', 'train_classification.csv'),
         val_csv=os.path.join('..', 'dataset', 'train_val_data', 'val_classification.csv'))
