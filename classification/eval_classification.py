from keras.models import load_model
import efficientnet.keras

import pandas as pd
import librosa as lr
import numpy as np
import cv2
import os
from functools import reduce
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm

from classification.labels import unique_labels


def mode(lst):
    """Calculates mode of a list

    @param lst:                                 list of values
    @return:                                    most common value in list
    """
    cnt = Counter(lst)
    return cnt.most_common(1)[0][0]


def calculate_acc(y_trues, y_preds):
    """Calculates accuracy for each slice and for each song

    @param y_trues:                             true values
    @param y_preds:                             predicted values
    """
    slice_acc = accuracy_score(reduce(lambda x, v: x + v, y_trues.values(), []),
                               reduce(lambda x, v: x + v, y_preds.values(), []))

    song_acc = accuracy_score(reduce(lambda x, v: x + [mode(v)], y_trues.values(), []),
                              reduce(lambda x, v: x + [mode(v)], y_preds.values(), []))

    print(f'Slice accuracy: {np.round(slice_acc, 4)}')
    print(f'Song accuracy: {np.round(song_acc, 4)}')


def conf_matrix(y_trues, y_preds):
    """Prints confusion matrix

    @param y_trues:                             true values
    @param y_preds:                             predicted values
    """
    print(confusion_matrix(reduce(lambda x, v: x + v, y_trues.values(), []),
                           reduce(lambda x, v: x + v, y_preds.values(), [])))


def report(y_trues, y_preds):
    """Prints classifiaction report

    @param y_trues:                             true values
    @param y_preds:                             predicted values
    """
    print(classification_report(reduce(lambda x, v: x + v, y_trues.values(), []),
                                reduce(lambda x, v: x + v, y_preds.values(), [])))


def main(val_csv, input_shape, model_paths):
    """Evaluate trained model

    @param val_csv:                             path to val .csv file
    @param input_shape:                         neural network input shape
    @param model_paths:                         path to saved model
    """
    val_df = pd.read_csv(val_csv)

    label_dict = {k: i for k, i in zip(unique_labels, [j for j in range(len(unique_labels))])}

    models = [load_model(model_path, compile=False) for model_path in model_paths]

    val_songs = list(val_df['song_name'].unique())

    y_trues = {k: [] for k in val_songs}
    y_preds = {k: [] for k in val_songs}
    for song in tqdm(val_songs):
        val_song_df = val_df[val_df['song_name'] == song].reset_index(drop=True)
        for i, row in val_song_df.iterrows():
            wav_path = row['wav_filename']
            slice, sr = lr.load(os.path.join('..', 'dataset', wav_path), sr=None)
            mel = lr.power_to_db(lr.feature.melspectrogram(y=slice, sr=sr), ref=np.max)

            x = np.expand_dims(cv2.resize(np.dstack([mel, mel, mel]), (input_shape[1], input_shape[0])), 0) / 255.

            p = np.argmax(np.mean([model.predict(x)[0] for model in models], axis=0))

            y = label_dict[row['artist']]

            y_trues[song].append(y)
            y_preds[song].append(p)

    calculate_acc(y_trues, y_preds)
    conf_matrix(y_trues, y_preds)
    report(y_trues, y_preds)


if __name__ == '__main__':
    main(val_csv=os.path.join('..', 'dataset', 'sample_data', 'val_classification.csv'),
         input_shape=(128, 256, 3),
         model_paths=[
                      os.path.join('models', 'mobilenetv2.h5'),
                      os.path.join('models', 'resnet50.h5'),
                      os.path.join('models', 'densenet121.h5'),
                      os.path.join('models', 'inceptionv3.h5'),
                    ])
