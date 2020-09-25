import argparse

from tools.tests import test_equality, test_labels
from tools.html2lrc import html2lrc
from tools.csv_creation import create_full_csv
from tools.csv_creation import create_stratified_openseq2seq_train_val_csv
from tools.csv_creation import create_stratified_classification_train_val_csv


def main(data_dir, dataset_type):
    """Creates datasets

    @param data_dir:                    path to directory containing all datasets
    """
    html2lrc(data_dir)
    test_equality(data_dir)
    test_labels(data_dir)
    create_full_csv(data_dir)

    if dataset_type == 'train_val':
        create_stratified_openseq2seq_train_val_csv(data_dir, 'dataset_all.csv')
        create_stratified_classification_train_val_csv(data_dir, 'dataset_all.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting the dataset directory')
    parser.add_argument('--data-dir', help='Directory to datasets', default='test_data')
    parser.add_argument('--dataset-type', help='Dataset type ("train_val" or "test")', default='test')
    args = parser.parse_args()

    main(data_dir=args.data_dir, dataset_type=args.dataset_type)
